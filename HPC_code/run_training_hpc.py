#!/usr/bin/env python3
"""
HPC_code.run_training_hpc

HPC entrypoint for the compute-heavy phases of the TDEW pipeline: it builds a Dask
cluster (``local`` | ``slurm`` | ``cuda``), injects the resulting ``Client`` into the
bucketed runners, and runs TRAIN-COEFFICIENTS and (optionally) FORECAST.

This is the SLURM/GPU analogue of ``Local/run_pipeline.sh``: it reuses the *same*
package functions (``run_bucketed_anomaly_training_dask`` /
``run_bucketed_forecast_dask``) via their injected-``client`` hook, so the algorithm
and outputs are identical — only the cluster (and therefore the parallel-processor
count ``P``) changes.

Prerequisite
------------
The bucketed inputs must already exist under ``--results`` (built once by
``Local/run_pipeline.sh`` prep steps, or any prior run): the prepared training
shards, the bucketed climatology, and — for forecasting — the bucketed future-TMIN
shards. This entrypoint deliberately runs only the parallel compute phases that the
scaling experiments measure; it does not rebuild I/O-bound inputs.

Examples
--------
Single-node baseline (P = 1), training only::

    python HPC_code/run_training_hpc.py \
        --base /data/henry_simcast_peru --results /data/henry_simcast_peru/results \
        --cluster local --n-workers 1

CPU SLURM strong-scaling point (P = 32), train + forecast::

    python HPC_code/run_training_hpc.py \
        --base /data/.../peru --results /data/.../results \
        --cluster slurm --slurm-queue cpu --slurm-account MYACCT \
        --slurm-cores 16 --slurm-memory 64GB --slurm-walltime 02:00:00 \
        --n-workers 32 --forecast

A100 GPU run (P = 4 GPUs), training only::

    python HPC_code/run_training_hpc.py \
        --base /data/.../peru --results /data/.../results \
        --cluster cuda --n-workers 4 --rmm-pool-size 35GB
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow `from hpc import ...` (sibling module) and `import tdew_estimation` (repo root)
# regardless of the caller's cwd or whether the package is pip-installed.
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
from tdew_estimation.forecast import (  # noqa: E402
    DaskForecastConfig,
    run_bucketed_forecast_dask,
)


def _parse_bucket_ids(spec: str | None) -> list[int] | None:
    """Parse ``--bucket-ids`` as either ``a,b,c`` or an inclusive ``start:end`` range."""
    if not spec:
        return None
    spec = spec.strip()
    if ":" in spec:
        start, end = spec.split(":", 1)
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run bucketed TDEW training (+ optional forecast) on an HPC Dask cluster.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Paths
    p.add_argument("--base", required=True, type=Path, help="Base path with variable folders/Outputs.")
    p.add_argument("--results", required=True, type=Path, help="Results dir holding bucketed inputs/outputs.")

    # Bucketed dataset layout (defaults match Local/run_pipeline.sh)
    p.add_argument("--prepared-dir", default="bucketed_training_data")
    p.add_argument("--clim-buckets-dir", default="climatology_by_bucket")
    p.add_argument("--coeffs-dir", default="llr_coeffs_anomaly_dataset")
    p.add_argument("--failures-dir", default="anomaly_failures")
    p.add_argument("--future-tmin-dir", default="future_tmin_by_bucket")
    p.add_argument("--predictions-dir", default="predictions")

    # Training config
    p.add_argument("--td-var", default="td")
    p.add_argument("--tmin-var", default="tmin_v1")
    p.add_argument("--train-start", type=int, default=1981)
    p.add_argument("--train-end", type=int, default=2016)
    p.add_argument("--h", type=int, default=11)
    p.add_argument("--kernel", default="Tricube", choices=["Tricube", "Gaussian"])
    p.add_argument("--min-samples", type=int, default=15)

    # Forecast config
    p.add_argument("--forecast", action="store_true", help="Also run the bucketed forecast phase.")
    p.add_argument("--pred-start", type=int, default=2017)
    p.add_argument("--pred-end", type=int, default=2020)
    p.add_argument("--history-end", type=int, default=None, help="Defaults to --train-end.")

    # Subsetting (weak scaling / smoke tests)
    p.add_argument(
        "--bucket-ids",
        default=None,
        help="Restrict to a subset of buckets: 'a,b,c' or inclusive range 'start:end'. "
        "Used for weak-scaling (n = n0*p) and smoke tests.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")

    # Cluster selection
    p.add_argument("--cluster", choices=["local", "slurm", "cuda"], default="local")
    p.add_argument("--n-workers", type=int, default=1, help="Worker processes (CPU) or GPUs (cuda). This is P.")
    p.add_argument("--threads", type=int, default=1, help="Threads per worker (keep 1 for BLAS-heavy fits).")
    p.add_argument("--batch-size", type=int, default=64, help="Max bucket tasks in flight (sliding window).")
    p.add_argument("--local-dir", default=None, help="Worker scratch/spill dir (node-local).")

    # SLURM-specific
    p.add_argument("--slurm-queue", default=None, help="SLURM partition (required for --cluster slurm).")
    p.add_argument("--slurm-account", default=None, help="SLURM billing account.")
    p.add_argument("--slurm-cores", type=int, default=16, help="Cores per SLURM job.")
    p.add_argument("--slurm-memory", default="64GB", help="Memory per SLURM job.")
    p.add_argument("--slurm-processes", type=int, default=None, help="Worker procs per job (default: = cores).")
    p.add_argument("--slurm-walltime", default="02:00:00")
    p.add_argument("--slurm-interface", default=None, help="Network interface (e.g. ib0).")
    p.add_argument("--slurm-log-dir", default=None)

    # CUDA-specific
    p.add_argument("--rmm-pool-size", default=None, help="Per-GPU RMM pool (e.g. 35GB).")
    p.add_argument("--device-memory-limit", default=None, help="Per-GPU spill threshold (e.g. 38GB).")
    p.add_argument("--cuda-visible-devices", default=None, help="Explicit GPU ordinals, e.g. '0,1,2,3'.")

    return p


def make_client(args: argparse.Namespace):
    """Build the requested cluster and return (client, label) for logging."""
    if args.cluster == "slurm":
        if not args.slurm_queue:
            raise SystemExit("--slurm-queue is required for --cluster slurm")
        client = make_slurm_cluster(
            cores=args.slurm_cores,
            memory=args.slurm_memory,
            queue=args.slurm_queue,
            account=args.slurm_account,
            processes=args.slurm_processes,
            walltime=args.slurm_walltime,
            n_workers=args.n_workers,
            local_directory=args.local_dir,
            log_directory=args.slurm_log_dir,
            interface=args.slurm_interface,
        )
        return client, f"slurm(P={args.n_workers})"
    if args.cluster == "cuda":
        client = make_local_cuda_cluster(
            n_workers=args.n_workers,
            cuda_visible_devices=args.cuda_visible_devices,
            rmm_pool_size=args.rmm_pool_size,
            device_memory_limit=args.device_memory_limit,
            local_directory=args.local_dir,
        )
        return client, f"cuda(P={args.n_workers} GPUs)"
    client = make_local_cpu_cluster(
        n_workers=args.n_workers,
        threads_per_worker=args.threads,
        local_directory=args.local_dir,
    )
    return client, f"local(P={args.n_workers})"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = args.results
    bucket_ids = _parse_bucket_ids(args.bucket_ids)
    history_end = args.history_end if args.history_end is not None else args.train_end

    prepared_root = results / args.prepared_dir
    clim_root = results / args.clim_buckets_dir
    coeffs_root = results / args.coeffs_dir
    failures_root = results / args.failures_dir
    future_tmin_root = results / args.future_tmin_dir
    predictions_root = results / args.predictions_dir

    train_cfg = AnomalyTrainingConfig(
        base_path=args.base,
        td_var=args.td_var,
        tmin_var=args.tmin_var,
        train_year_range=(args.train_start, args.train_end),
        h=args.h,
        kernel=args.kernel,
        min_samples=args.min_samples,
    )

    client, label = make_client(args)
    dashboard = getattr(client, "dashboard_link", "n/a")
    print(f"[hpc] cluster={label} dashboard={dashboard}", flush=True)
    if bucket_ids is not None:
        print(f"[hpc] restricted to {len(bucket_ids)} buckets", flush=True)

    try:
        # --- TRAIN-COEFFICIENTS (primary benchmarked phase) ---
        t0 = time.perf_counter()
        train_summaries = run_bucketed_anomaly_training_dask(
            prepared_training_root=prepared_root,
            bucketed_climatology_root=clim_root,
            coeffs_output_root=coeffs_root,
            config=train_cfg,
            bucket_ids=bucket_ids,
            dask_config=DaskAnomalyConfig(
                n_workers=args.n_workers,
                threads_per_worker=args.threads,
                batch_size=args.batch_size,
                local_directory=args.local_dir,
            ),
            failure_output_root=failures_root,
            overwrite=args.overwrite,
            client=client,
        )
        train_s = time.perf_counter() - t0
        coeff_rows = sum(s.coeff_rows for s in train_summaries)
        fail_rows = sum(s.failure_rows for s in train_summaries)
        print(
            f"[hpc] TRAIN done in {train_s:.1f}s | buckets={len(train_summaries)} "
            f"coeff_rows={coeff_rows} failure_rows={fail_rows}",
            flush=True,
        )

        # --- FORECAST (optional) ---
        if args.forecast:
            t0 = time.perf_counter()
            fc_summaries = run_bucketed_forecast_dask(
                coeffs_root=coeffs_root,
                climatology_root=clim_root,
                prepared_training_root=prepared_root,
                future_tmin_root=future_tmin_root,
                predictions_output_root=predictions_root,
                prediction_years=(args.pred_start, args.pred_end),
                history_end_year=history_end,
                bucket_ids=bucket_ids,
                dask_config=DaskForecastConfig(
                    n_workers=args.n_workers,
                    threads_per_worker=args.threads,
                    batch_size=args.batch_size,
                    local_directory=args.local_dir,
                ),
                overwrite=args.overwrite,
                client=client,
            )
            fc_s = time.perf_counter() - t0
            print(
                f"[hpc] FORECAST done in {fc_s:.1f}s | buckets={len(fc_summaries)}",
                flush=True,
            )
    finally:
        cluster = getattr(client, "cluster", None)
        client.close()
        if cluster is not None:
            try:
                cluster.close()
            except Exception:  # pragma: no cover - best-effort teardown
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
