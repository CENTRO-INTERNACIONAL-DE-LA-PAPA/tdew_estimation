"""
HPC_code.hpc

Dask cluster builders for the TDEW estimation pipeline on HPC.

The training and forecast runners in :mod:`tdew_estimation.anomaly_dask` and
:mod:`tdew_estimation.forecast` accept an injected ``client`` and leave it open
for the caller. This module provides the two clusters the scaling experiments
need, both returning a live ``distributed.Client``:

- :func:`make_slurm_cluster` — a multi-node CPU cluster via ``dask_jobqueue.SLURMCluster``.
  One single-threaded worker process per core (BLAS pinned to 1) is the unit of the
  parallel-processor axis ``P`` in the experimental methodology.
- :func:`make_local_cuda_cluster` — a single-node multi-GPU cluster via ``dask_cuda.LocalCUDACluster``.
  One worker per A100; here ``P`` = number of GPUs.

Both are optional dependencies (``dask-jobqueue`` for SLURM, ``dask-cuda`` for GPU);
they are imported lazily so the CPU-only / local code paths keep working without them.

The definition of ``P`` is kept identical to the document: a worker process is one
unit of ``P``, whether it drives a CPU core or a single A100.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence


def _pin_blas_threads_in_env(env: dict[str, str]) -> dict[str, str]:
    """
    Ensure BLAS/OpenMP thread pools are pinned to a single thread inside worker jobs.

    Mirrors ``tdew_estimation.anomaly_dask._configure_blas_threads`` but injects the
    values into the *job* environment so SLURM-spawned worker processes inherit them.
    An explicit value already present in ``env`` wins (``setdefault`` semantics).
    """
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env.setdefault(var, "1")
    return env


def make_slurm_cluster(
    *,
    cores: int,
    memory: str,
    queue: str,
    account: Optional[str] = None,
    processes: Optional[int] = None,
    walltime: str = "02:00:00",
    n_workers: int = 0,
    minimum_jobs: Optional[int] = None,
    maximum_jobs: Optional[int] = None,
    local_directory: Optional[str] = None,
    log_directory: Optional[str] = None,
    interface: Optional[str] = None,
    death_timeout: int = 120,
    job_extra_directives: Optional[Sequence[str]] = None,
    job_script_prologue: Optional[Sequence[str]] = None,
    scheduler_timeout_s: int = 120,
) -> Any:
    """
    Build a ``dask_jobqueue.SLURMCluster`` and return a connected ``Client``.

    Each SLURM job provides ``cores`` cores and ``memory`` RAM, split into
    ``processes`` single-threaded worker processes (default: one process per core,
    ``threads_per_worker = cores / processes = 1``). This keeps every worker a
    one-BLAS-thread process so the per-(ID,DOY) WLS fits do not oversubscribe cores.

    Scaling: the number of parallel processors ``P`` is ``processes * (#jobs)``. Use
    ``n_workers`` for a fixed fleet (``cluster.scale(n_workers)``) or
    ``minimum_jobs``/``maximum_jobs`` for adaptive scaling.

    Parameters
    ----------
    cores, memory, queue, account, walltime:
        Per-job SLURM resources. ``queue`` is the partition; ``account`` the billing
        account (``--account``). Provide both from the cluster's docs.
    processes:
        Worker processes per job. Defaults to ``cores`` (one single-threaded worker
        per core).
    n_workers:
        If > 0, immediately ``scale`` to this many worker processes (fixed fleet).
    minimum_jobs, maximum_jobs:
        If both set, use ``adapt(minimum_jobs=..., maximum_jobs=...)`` instead.
    local_directory:
        Worker scratch/spill dir; point at node-local storage (e.g. ``$TMPDIR``), not
        a networked filesystem.
    log_directory:
        Where SLURM writes per-job worker logs.
    interface:
        Network interface for scheduler<->worker comms (e.g. ``ib0`` for InfiniBand).
    job_extra_directives:
        Extra ``#SBATCH`` lines (e.g. ``["--exclusive"]``).
    job_script_prologue:
        Shell lines run at the top of each job script (e.g. ``module load`` / venv
        activation). BLAS-pinning exports are added automatically.

    Returns
    -------
    distributed.Client
        Connected to the new ``SLURMCluster``. The caller owns it and must close both
        the client and ``client.cluster`` when done.
    """
    try:
        from dask_jobqueue import SLURMCluster
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "make_slurm_cluster requires 'dask-jobqueue'. Install it with "
            "`pip install dask-jobqueue` (it is listed in HPC_code/requirements-gpu.txt "
            "and the project's hpc extra)."
        ) from exc
    from dask.distributed import Client

    procs = int(processes) if processes else int(cores)
    if procs < 1:
        raise ValueError("processes must be >= 1")

    job_extra = list(job_extra_directives) if job_extra_directives else []
    if account:
        # dask-jobqueue passes `account` through, but keep an explicit directive too
        # for schedulers that need --account on the worker jobs.
        pass

    env_exports = _pin_blas_threads_in_env({})
    prologue = list(job_script_prologue) if job_script_prologue else []
    prologue = [f"export {k}={v}" for k, v in env_exports.items()] + prologue

    kwargs: dict[str, Any] = dict(
        cores=int(cores),
        processes=procs,
        memory=memory,
        queue=queue,
        walltime=walltime,
        job_extra_directives=job_extra,
        job_script_prologue=prologue,
        death_timeout=death_timeout,
    )
    if account:
        kwargs["account"] = account
    if local_directory:
        kwargs["local_directory"] = local_directory
    if log_directory:
        kwargs["log_directory"] = log_directory
    if interface:
        kwargs["interface"] = interface

    cluster = SLURMCluster(**kwargs)

    if minimum_jobs is not None and maximum_jobs is not None:
        cluster.adapt(minimum_jobs=int(minimum_jobs), maximum_jobs=int(maximum_jobs))
    elif n_workers and n_workers > 0:
        cluster.scale(n=int(n_workers))

    client = Client(cluster, timeout=f"{scheduler_timeout_s}s")
    return client


def make_local_cuda_cluster(
    *,
    n_workers: Optional[int] = None,
    cuda_visible_devices: Optional[str] = None,
    rmm_pool_size: Optional[str] = None,
    device_memory_limit: Optional[str] = None,
    local_directory: Optional[str] = None,
    threads_per_worker: int = 1,
    dashboard_address: Optional[str] = ":8787",
    scheduler_timeout_s: int = 120,
) -> Any:
    """
    Build a ``dask_cuda.LocalCUDACluster`` (one worker per GPU) and return a ``Client``.

    On a single A100 node this gives one Dask worker per visible GPU, so the
    parallel-processor axis ``P`` equals the number of GPUs. Within each GPU the
    batched WLS solver (Phase 3) uses all CUDA cores; that intra-GPU parallelism is a
    fixed per-fit constant folded into the model, exactly as the methodology states.

    Parameters
    ----------
    n_workers:
        Number of GPU workers. Defaults to all visible GPUs. Set this (or
        ``cuda_visible_devices``) to vary ``P`` in the GPU strong-scaling sweep.
    cuda_visible_devices:
        Comma-separated GPU ordinals (e.g. ``"0,1,2,3"``). Selects which GPUs to use.
    rmm_pool_size:
        Pre-allocated RMM memory pool per worker (e.g. ``"35GB"``). Stabilizes timing
        by avoiding repeated cudaMalloc.
    device_memory_limit:
        Per-worker device-memory spill threshold (e.g. ``"38GB"``).
    local_directory:
        Host scratch/spill dir (node-local).

    Returns
    -------
    distributed.Client
        Connected to the new ``LocalCUDACluster``. The caller owns it.
    """
    try:
        from dask_cuda import LocalCUDACluster
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "make_local_cuda_cluster requires 'dask-cuda'. Install the GPU extras "
            "from HPC_code/requirements-gpu.txt (cu12 wheels)."
        ) from exc
    from dask.distributed import Client

    kwargs: dict[str, Any] = dict(threads_per_worker=int(threads_per_worker))
    if cuda_visible_devices is not None:
        kwargs["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    elif n_workers is not None:
        # Restrict to the first n_workers GPUs deterministically.
        kwargs["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(int(n_workers)))
    if rmm_pool_size:
        kwargs["rmm_pool_size"] = rmm_pool_size
    if device_memory_limit:
        kwargs["device_memory_limit"] = device_memory_limit
    if local_directory:
        kwargs["local_directory"] = local_directory
    if dashboard_address is not None:
        kwargs["dashboard_address"] = dashboard_address

    cluster = LocalCUDACluster(**kwargs)
    client = Client(cluster, timeout=f"{scheduler_timeout_s}s")
    return client


def make_local_cpu_cluster(
    *,
    n_workers: int,
    threads_per_worker: int = 1,
    memory_limit: str = "auto",
    local_directory: Optional[str] = None,
    dashboard_address: Optional[str] = ":8787",
    scheduler_timeout_s: int = 120,
) -> Any:
    """
    Build a plain single-node ``distributed.LocalCluster`` and return a ``Client``.

    This is the dev-box / single-SLURM-node fallback (and the baseline ``P = 1`` run
    for scaling). It mirrors ``tdew_estimation.anomaly_dask._make_client`` but is
    exposed here so the HPC entrypoint can request ``local|slurm|cuda`` uniformly and
    always inject a client into the runners.
    """
    from dask.distributed import Client, LocalCluster

    _pin_blas_threads_in_env(os.environ)  # type: ignore[arg-type]
    kwargs: dict[str, Any] = dict(
        n_workers=int(n_workers),
        threads_per_worker=int(threads_per_worker),
        memory_limit=memory_limit,
    )
    if local_directory:
        kwargs["local_directory"] = local_directory
    if dashboard_address is not None:
        kwargs["dashboard_address"] = dashboard_address
    cluster = LocalCluster(**kwargs)
    return Client(cluster)
