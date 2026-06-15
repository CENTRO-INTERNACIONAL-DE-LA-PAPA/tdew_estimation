#!/usr/bin/env python3
"""
HPC_code.benchmark_scaling

Driver for the D4 strong/weak scaling experiments. For each processor count ``p`` it
builds a *fresh* Dask cluster sized to ``p``, times the bucketed compute phase(s) by
injecting the live ``Client`` into the *same* runners the production entrypoint uses
(:func:`run_bucketed_anomaly_training_dask` / :func:`run_bucketed_forecast_dask`), and
appends one row per (phase, trial) to a CSV. The CSV feeds ``analyze_scaling.py``,
which computes speedup ``S(p)`` / efficiency ``E(p)`` and renders the PRAM tables/plots.

The cluster is selected with ``--cluster {local,slurm,cuda}``: ``local`` builds a
single-node ``LocalCluster`` sized to ``p`` (dev box / one fat node), ``slurm`` builds a
multi-node ``dask-jobqueue`` fleet of ``p`` worker processes (the D6 CPU path), and
``cuda`` builds a single-node ``dask-cuda`` cluster of ``p`` GPU workers (the D6 GPU path).
For ``slurm``/``cuda`` the harness waits for the fleet to come up
(``client.wait_for_workers(p)``) before starting the timer, so queue/startup latency is
excluded from ``wall_s``.

``--hw gpu`` routes the TRAIN phase to the GPU batched-WLS trainer
(``HPC_code/gpu_train.py``); pair it with ``--cluster cuda`` for real multi-GPU scaling.
Without ``--cluster cuda`` the GPU trainer runs sequentially on the single local GPU
(dev-box smoke). Forecast stays on the CPU runner. The ``dask-cuda`` cluster requires a
RAPIDS env (HPC); the single-GPU path needs only CuPy.

CSV schema (10 columns, appended incrementally so partial runs survive)::

    dataset, mode, hw, p, n_ids, B, phase, trial, wall_s, timestamp

Examples
--------
Synthetic CPU strong-scaling smoke::

    python HPC_code/benchmark_scaling.py \
        --base /tmp/tdew_synth/base --results /tmp/tdew_synth/results \
        --hw cpu --mode strong --p-list 1,2,4 --trials 1 --num-buckets 8 \
        --phases train --dataset-label synth --synth \
        --out-csv /tmp/tdew_synth/scaling.csv

Real CPU strong-scaling point on a SLURM node (inputs prebuilt under --results)::

    python HPC_code/benchmark_scaling.py \
        --base /data/.../peru --results /data/.../results \
        --hw cpu --mode strong --p-list 1,2,4,8,16,32 --trials 3 \
        --num-buckets 256 --phases train,forecast --dataset-label v1 \
        --out-csv results/scaling_cpu_v1.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

# Allow `from hpc import ...` and `import tdew_estimation` / `import _synth`.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from hpc import (  # noqa: E402
    make_local_cpu_cluster,
    make_local_cuda_cluster,
    make_slurm_cluster,
)

from tdew_estimation.anomaly_dask import (  # noqa: E402
    DaskAnomalyConfig,
    run_bucketed_anomaly_training_dask,
)
from tdew_estimation.anomaly_train import AnomalyTrainingConfig  # noqa: E402
from tdew_estimation.bucket_layout import bucket_dir, discover_bucket_ids  # noqa: E402
from tdew_estimation.forecast import (  # noqa: E402
    DaskForecastConfig,
    run_bucketed_forecast_dask,
)

log = logging.getLogger("benchmark_scaling")

CSV_COLUMNS = [
    "dataset",
    "mode",
    "hw",
    "p",
    "n_ids",
    "B",
    "phase",
    "trial",
    "wall_s",
    "timestamp",
]


def _parse_int_list(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def make_bench_client(args: argparse.Namespace, p: int):
    """Build the cluster sized to ``p`` for the requested ``--cluster`` backend.

    Returns a live ``Client``. For ``slurm`` the caller should
    ``client.wait_for_workers(p, ...)`` before timing so queue/startup latency is not
    counted as compute time.
    """
    if args.cluster == "slurm":
        if not args.slurm_queue:
            raise SystemExit("--slurm-queue is required for --cluster slurm")
        return make_slurm_cluster(
            cores=args.slurm_cores,
            memory=args.slurm_memory,
            queue=args.slurm_queue,
            account=args.slurm_account,
            processes=args.slurm_processes,
            walltime=args.slurm_walltime,
            n_workers=p,
            local_directory=args.local_dir,
            log_directory=args.slurm_log_dir,
            interface=args.slurm_interface,
        )
    if args.cluster == "cuda":
        return make_local_cuda_cluster(
            n_workers=p,
            cuda_visible_devices=args.cuda_visible_devices,
            rmm_pool_size=args.rmm_pool_size,
            device_memory_limit=args.device_memory_limit,
            local_directory=args.local_dir,
        )
    return make_local_cpu_cluster(
        n_workers=p,
        threads_per_worker=1,
        local_directory=args.local_dir,
    )


def _close_client(client) -> None:
    """Tear down a client and its cluster (mirrors run_training_hpc.py finally block)."""
    cluster = getattr(client, "cluster", None)
    try:
        client.close()
    finally:
        if cluster is not None:
            try:
                cluster.close()
            except Exception:  # pragma: no cover - best-effort teardown
                pass


def count_ids(prepared_root: Path, bucket_ids: list[int]) -> int:
    """Distinct ``ID`` count across the prepared shards of the selected buckets."""
    ids: set[int] = set()
    for bid in bucket_ids:
        bdir = bucket_dir(prepared_root, int(bid))
        if not bdir.exists():
            continue
        for f in sorted(bdir.glob("train_*.parquet")):
            table = pq.read_table(f, columns=["ID"])
            ids.update(table.column("ID").to_pylist())
    return len(ids)


def select_buckets(
    all_bucket_ids: list[int],
    *,
    mode: str,
    p: int,
    num_buckets: int,
    n0: int,
    p_max: int,
) -> list[int]:
    """Pick the bucket subset for a given (mode, p).

    strong: one fixed set for every p — the first ``num_buckets`` discovered buckets.
    weak:   n(p) = n0 * p — the first ``n0 * p`` discovered buckets.
    """
    if mode == "strong":
        want = num_buckets
        if want > len(all_bucket_ids):
            log.warning(
                "strong: requested B=%s but only %s buckets exist; using all.",
                want,
                len(all_bucket_ids),
            )
        chosen = all_bucket_ids[:want]
        if len(chosen) < 4 * p_max:
            log.warning(
                "strong: B=%s < 4*max(p)=%s — too few buckets for clean load balance "
                "at the largest p (smoke runs only).",
                len(chosen),
                4 * p_max,
            )
        return chosen

    # weak
    want = n0 * p
    if want > len(all_bucket_ids):
        log.warning(
            "weak: need n0*p=%s buckets for p=%s but only %s exist; using all.",
            want,
            p,
            len(all_bucket_ids),
        )
    return all_bucket_ids[:want]


def _append_rows(csv_path: Path, rows: list[dict]) -> None:
    """Append rows to the CSV (write header if the file is new), flushing each call."""
    new_file = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
        fh.flush()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Strong/weak scaling benchmark for bucketed TDEW training/forecast.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base", required=True, type=Path)
    p.add_argument("--results", required=True, type=Path)

    p.add_argument("--hw", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--mode", choices=["strong", "weak"], default="strong")
    p.add_argument("--p-list", default="1,2,4,8", help="Processor counts, e.g. '1,2,4,8'.")
    p.add_argument("--trials", type=int, default=3, help="Repeats per p (median is used).")
    p.add_argument("--num-buckets", type=int, default=64, help="B for strong mode.")
    p.add_argument("--n0", type=int, default=4, help="Weak-scaling base buckets per proc.")
    p.add_argument("--phases", default="train", help="Comma list: train[,forecast].")
    p.add_argument("--dataset-label", default="v1")
    p.add_argument("--out-csv", required=True, type=Path)
    p.add_argument("--synth", action="store_true", help="Build a synthetic dataset first.")

    # Bucketed dataset layout (defaults match Local/run_pipeline.sh & run_training_hpc.py)
    p.add_argument("--prepared-dir", default="bucketed_training_data")
    p.add_argument("--clim-buckets-dir", default="climatology_by_bucket")
    p.add_argument("--coeffs-dir", default="llr_coeffs_anomaly_dataset")
    p.add_argument("--failures-dir", default="anomaly_failures")
    p.add_argument("--future-tmin-dir", default="future_tmin_by_bucket")
    p.add_argument("--predictions-dir", default="predictions")

    # Training config passthrough
    p.add_argument("--td-var", default="td")
    p.add_argument("--tmin-var", default="tmin_v1")
    p.add_argument("--train-start", type=int, default=1981)
    p.add_argument("--train-end", type=int, default=2016)
    p.add_argument("--h", type=int, default=11)
    p.add_argument("--kernel", default="Tricube", choices=["Tricube", "Gaussian"])
    p.add_argument("--min-samples", type=int, default=15)

    # Forecast config
    p.add_argument("--pred-start", type=int, default=2017)
    p.add_argument("--pred-end", type=int, default=2020)
    p.add_argument("--history-end", type=int, default=None, help="Defaults to --train-end.")

    # Cluster backend
    p.add_argument(
        "--cluster",
        choices=["local", "slurm", "cuda"],
        default="local",
        help="local = single-node LocalCluster sized to p; slurm = dask-jobqueue fleet; "
        "cuda = single-node dask-cuda cluster of p GPU workers (D6 GPU scaling).",
    )
    p.add_argument(
        "--gpu-train",
        action="store_true",
        help="Route the TRAIN phase to the GPU batched-WLS trainer. Implied by --hw gpu. "
        "With --cluster cuda it distributes over p GPUs; otherwise it runs on the single "
        "local GPU (client-agnostic). Forecast always stays on the CPU runner.",
    )
    p.add_argument(
        "--gpu-block",
        type=int,
        default=128,
        help="CUDA threads-per-block for the GPU kernel (occupancy tuning; default 128).",
    )
    p.add_argument(
        "--worker-timeout",
        type=int,
        default=1800,
        help="Seconds to wait for the fleet to come up before timing (slurm/cuda).",
    )

    # CUDA-specific (used with --cluster cuda)
    p.add_argument("--rmm-pool-size", default=None, help="Per-GPU RMM pool (e.g. 35GB).")
    p.add_argument("--device-memory-limit", default=None, help="Per-GPU spill threshold (e.g. 38GB).")
    p.add_argument("--cuda-visible-devices", default=None, help="Explicit GPU ordinals, e.g. '0,1,2,3'.")

    # SLURM-specific (used with --cluster slurm)
    p.add_argument("--slurm-queue", default=None, help="SLURM partition (required for slurm).")
    p.add_argument("--slurm-account", default=None, help="SLURM billing account.")
    p.add_argument("--slurm-cores", type=int, default=16, help="Cores per SLURM job.")
    p.add_argument("--slurm-memory", default="64GB", help="Memory per SLURM job.")
    p.add_argument("--slurm-processes", type=int, default=None, help="Procs per job (default: =cores).")
    p.add_argument("--slurm-walltime", default="02:00:00")
    p.add_argument("--slurm-interface", default=None, help="Network interface (e.g. ib0).")
    p.add_argument("--slurm-log-dir", default=None)

    # Dask knobs
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--local-dir", default=None)

    # Synthetic sizing (only used with --synth)
    p.add_argument("--synth-ids", type=int, default=40)
    p.add_argument("--synth-seed", type=int, default=1234)
    return p


def run_phase(
    phase: str,
    *,
    args: argparse.Namespace,
    client,
    p: int,
    bucket_ids: list[int],
    train_cfg: AnomalyTrainingConfig,
    roots: dict[str, Path],
    history_end: int,
    use_gpu_train: bool = False,
) -> float:
    """Time a single phase via the injected-client runner. Returns wall seconds.

    The runner reuses the injected ``client``, so ``dask_config`` only tunes the
    sliding-window size (``max(batch_size, n_workers)``); we set ``n_workers=p`` to
    match the live cluster.

    When ``use_gpu_train`` the TRAIN phase runs the GPU batched-WLS trainer instead;
    it distributes over the cuda ``client`` (``--cluster cuda``), or runs sequentially
    on the single local GPU (``client=None``) for any other cluster.
    """
    if phase == "train" and use_gpu_train:
        from gpu_train import run_bucketed_anomaly_training_gpu  # lazy: needs CuPy

        gpu_client = client if args.cluster == "cuda" else None
        t0 = time.perf_counter()
        run_bucketed_anomaly_training_gpu(
            prepared_training_root=roots["prepared"],
            bucketed_climatology_root=roots["clim"],
            coeffs_output_root=roots["coeffs"],
            config=train_cfg,
            bucket_ids=bucket_ids,
            failure_output_root=roots["failures"],
            overwrite=True,
            client=gpu_client,
            block=args.gpu_block,
            max_in_flight=args.batch_size,
        )
        return time.perf_counter() - t0

    if phase == "train":
        t0 = time.perf_counter()
        run_bucketed_anomaly_training_dask(
            prepared_training_root=roots["prepared"],
            bucketed_climatology_root=roots["clim"],
            coeffs_output_root=roots["coeffs"],
            config=train_cfg,
            bucket_ids=bucket_ids,
            dask_config=DaskAnomalyConfig(
                n_workers=p,
                threads_per_worker=1,
                batch_size=args.batch_size,
                local_directory=args.local_dir,
            ),
            failure_output_root=roots["failures"],
            overwrite=True,
            client=client,
        )
        return time.perf_counter() - t0

    if phase == "forecast":
        t0 = time.perf_counter()
        run_bucketed_forecast_dask(
            coeffs_root=roots["coeffs"],
            climatology_root=roots["clim"],
            prepared_training_root=roots["prepared"],
            future_tmin_root=roots["future_tmin"],
            predictions_output_root=roots["predictions"],
            prediction_years=(args.pred_start, args.pred_end),
            history_end_year=history_end,
            bucket_ids=bucket_ids,
            dask_config=DaskForecastConfig(
                n_workers=p,
                threads_per_worker=1,
                batch_size=args.batch_size,
                local_directory=args.local_dir,
            ),
            overwrite=True,
            client=client,
        )
        return time.perf_counter() - t0

    raise ValueError(f"Unknown phase: {phase!r}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)

    # --hw gpu routes TRAIN to the GPU batched-WLS trainer (D3). --gpu-train forces it
    # explicitly; either implies GPU training for the train phase below.
    use_gpu_train = args.gpu_train or args.hw == "gpu"
    if use_gpu_train and "forecast" in args.phases:
        log.warning(
            "GPU training is enabled but the forecast phase stays on the CPU runner "
            "(GPU forecast is out of scope)."
        )

    p_list = _parse_int_list(args.p_list)
    p_max = max(p_list)
    phases = [s.strip() for s in args.phases.split(",") if s.strip()]
    history_end = args.history_end if args.history_end is not None else args.train_end

    if args.synth:
        import _synth

        log.info("[synth] building synthetic dataset (ids=%s)", args.synth_ids)
        _synth.build_synthetic_results(
            args.base,
            args.results,
            n_ids=args.synth_ids,
            year_range=(args.train_start, args.train_end),
            num_buckets=args.num_buckets,
            seed=args.synth_seed,
            td_var=args.td_var,
            tmin_var=args.tmin_var,
            with_forecast=("forecast" in phases),
            pred_years=(args.pred_start, args.pred_end),
        )

    roots = {
        "prepared": args.results / args.prepared_dir,
        "clim": args.results / args.clim_buckets_dir,
        "coeffs": args.results / args.coeffs_dir,
        "failures": args.results / args.failures_dir,
        "future_tmin": args.results / args.future_tmin_dir,
        "predictions": args.results / args.predictions_dir,
    }

    all_bucket_ids = discover_bucket_ids(roots["prepared"])
    if not all_bucket_ids:
        raise SystemExit(
            f"No buckets found under {roots['prepared']}. Build inputs first "
            f"(use --synth, or run Local/run_pipeline.sh prep steps)."
        )
    log.info("[bench] discovered %s buckets under %s", len(all_bucket_ids), roots["prepared"])

    train_cfg = AnomalyTrainingConfig(
        base_path=args.base,
        td_var=args.td_var,
        tmin_var=args.tmin_var,
        train_year_range=(args.train_start, args.train_end),
        h=args.h,
        kernel=args.kernel,
        min_samples=args.min_samples,
    )

    # Cache n_ids per bucket-set signature to avoid recomputation across trials.
    n_ids_cache: dict[tuple[int, ...], int] = {}

    for p in p_list:
        bucket_ids = select_buckets(
            all_bucket_ids,
            mode=args.mode,
            p=p,
            num_buckets=args.num_buckets,
            n0=args.n0,
            p_max=p_max,
        )
        sig = tuple(bucket_ids)
        if sig not in n_ids_cache:
            n_ids_cache[sig] = count_ids(roots["prepared"], bucket_ids)
        n_ids = n_ids_cache[sig]
        B = len(bucket_ids)
        log.info("[bench] p=%s mode=%s B=%s n_ids=%s", p, args.mode, B, n_ids)

        # One cluster per p, reused across this p's trials. For SLURM this avoids paying
        # the queue wait `trials` times; the median over trials still absorbs warmup.
        client = make_bench_client(args, p)
        try:
            client.wait_for_workers(p, timeout=args.worker_timeout)
            for trial in range(1, args.trials + 1):
                rows = []
                for phase in phases:
                    wall_s = run_phase(
                        phase,
                        args=args,
                        client=client,
                        p=p,
                        bucket_ids=bucket_ids,
                        train_cfg=train_cfg,
                        roots=roots,
                        history_end=history_end,
                        use_gpu_train=use_gpu_train,
                    )
                    log.info(
                        "[bench]   p=%s trial=%s phase=%s wall_s=%.3f",
                        p,
                        trial,
                        phase,
                        wall_s,
                    )
                    rows.append(
                        {
                            "dataset": args.dataset_label,
                            "mode": args.mode,
                            "hw": args.hw,
                            "p": p,
                            "n_ids": n_ids,
                            "B": B,
                            "phase": phase,
                            "trial": trial,
                            "wall_s": round(wall_s, 6),
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                        }
                    )
                _append_rows(args.out_csv, rows)
        finally:
            _close_client(client)

    log.info("[bench] done -> %s", args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
