"""
ps_kinematics.gpu_manager — GPU resource management for multi-worker pipelines.

Prevents GPU out-of-memory and CUDA context thrashing when multiple
ProcessPoolExecutor workers share a single GPU for SuperRes / RTMPose.

Strategy:
  A multiprocessing.Semaphore limits GPU access to GPU_CONCURRENCY workers
  at a time (default 1).  Workers that don't hold the semaphore continue
  CPU-only work (MediaPipe tracking, fill passes, kinematics).
  After each GPU refinement pass the worker explicitly frees model weights
  and calls torch.cuda.empty_cache() so the next worker starts with a
  clean GPU.
"""

import logging
import os

logger = logging.getLogger(__name__)

# ── Module-level semaphore reference (set via worker initializer) ─────────
_gpu_semaphore = None
_gpu_held = False  # per-worker flag: True while this process holds the GPU semaphore

# ── Per-worker report queue (set via worker initializer) ─────────────────
# Workers put sentinel dicts here so the parent process can track semaphore
# ownership and recover it if the worker is OOM-killed or force-killed while
# holding the GPU lock.  None when not running under the parallel pipeline.
_report_queue = None
_worker_pid = None


def init_gpu_semaphore(sem, report_queue=None):
    """Worker initializer callback: store the shared GPU semaphore.

    Parameters
    ----------
    sem : multiprocessing.Semaphore
        Shared semaphore limiting concurrent GPU access.
    report_queue : multiprocessing.Queue or None
        Per-worker queue used to send GPU-acquisition sentinels back to the
        parent so it can recover the semaphore if this worker is killed.
    """
    global _gpu_semaphore, _report_queue, _worker_pid
    _gpu_semaphore = sem
    _report_queue = report_queue
    _worker_pid = os.getpid()


# How long (seconds) to wait for the GPU semaphore before giving up.
# If a worker process is killed while holding the lock (OOM, SIGKILL, etc.)
# the count is never restored and all waiting workers would hang forever
# without this timeout.
_GPU_ACQUIRE_TIMEOUT_S = 300  # 5 minutes


def acquire_gpu(timeout=None) -> bool:
    """Block until exclusive GPU access is available.

    Parameters
    ----------
    timeout : float or None
        Seconds to wait before giving up.  None uses the module default
        (``_GPU_ACQUIRE_TIMEOUT_S``).  Pass 0 for a non-blocking poll.

    Returns
    -------
    bool
        True if the semaphore was successfully acquired.
        False if the timeout expired (caller must NOT call release_gpu).
    """
    global _gpu_held
    if _gpu_semaphore is None:
        _gpu_held = True
        return True  # no semaphore in sequential mode — always "acquired"
    if timeout is None:
        timeout = _GPU_ACQUIRE_TIMEOUT_S
    acquired = _gpu_semaphore.acquire(timeout=timeout)
    if not acquired:
        logger.warning(
            "acquire_gpu() timed out after %ss (pid=%s). "
            "A worker likely died while holding the GPU semaphore. "
            "GPU refinement will be skipped for this video to avoid a permanent stall.",
            timeout,
            os.getpid(),
        )
        return False
    _gpu_held = True
    # Signal the parent that this worker now holds the semaphore so it can
    # recover it if this process is OOM-killed or force-killed.
    if _report_queue is not None:
        try:
            _report_queue.put_nowait({"_gpu_acquired": True, "pid": _worker_pid})
        except Exception:
            pass
    return True


def release_gpu():
    """Release GPU access so the next waiting worker can proceed.

    Safe to call even if the semaphore was never acquired (e.g. after a
    timeout or in a ``finally`` block) — the per-worker ``_gpu_held`` flag
    prevents double-release from incrementing the semaphore above its
    initial count.
    """
    global _gpu_held
    if not _gpu_held:
        return  # nothing to release — prevents double-release / spurious release
    _gpu_held = False
    # Signal the parent BEFORE releasing so there is no window in which the
    # semaphore is free but the parent still thinks we hold it.
    if _report_queue is not None:
        try:
            _report_queue.put_nowait({"_gpu_released": True, "pid": _worker_pid})
        except Exception:
            pass
    if _gpu_semaphore is not None:
        _gpu_semaphore.release()


def cleanup_gpu():
    """Free GPU model weights and empty the CUDA cache.

    Called after the GPU refinement pass finishes so the next worker
    starts with maximum available GPU memory.
    """
    try:
        from .refinement.superres import cleanup_superres

        cleanup_superres()
    except Exception:
        pass
    try:
        from .refinement.rtmpose import cleanup_rtmpose

        cleanup_rtmpose()
    except Exception:
        pass
    try:
        from .refinement.yolo import cleanup_yolo

        cleanup_yolo()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
