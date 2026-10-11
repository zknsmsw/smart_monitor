"""Optional CPU affinity and scheduling hints for multi-core optimisation.

Usage (for experimentation only — the default Linux scheduler usually does
a good job on its own)::

    from core.cpu_tune import pin_thread, set_nice

    pin_thread(os.getpid(), [0, 1, 2, 3])   # allow all cores
    pin_thread(0, [2, 3])                    # pin main thread to cores 2-3
    set_nice(-5)                             # slightly higher priority

CPU core recommendation for RK3566 (4× Cortex-A55):

    Core 0 — Linux housekeeping (interrupts, ksoftirqd, RCU callbacks)
    Core 1 — FaceWorker  (NPU inference + OpenCV + numpy — heavy CPU)
    Core 2 — Main loop   (motion detection + recording write + JPEG encode)
    Core 3 — Flask web server + camera capture thread + misc

Even without explicit pinning, the CFS scheduler will naturally distribute
threads across idle cores.  Pinning is only useful as a last resort if you
observe excessive context-switch overhead or unwanted migration.
"""
import os
import logging

log = logging.getLogger(__name__)

try:
    _HAVE_SCHED = hasattr(os, 'sched_setaffinity')
except Exception:
    _HAVE_SCHED = False


def pin_thread(tid, cores):
    """Pin a thread (by tid) to the given CPU core list.

    ``tid=0`` means the calling thread.  ``cores`` is a list of integer
    core indices, e.g. ``[2, 3]``.

    Returns True on success, False if the platform doesn't support it.
    """
    if not _HAVE_SCHED:
        log.debug("os.sched_setaffinity not available on this platform")
        return False
    try:
        os.sched_setaffinity(tid, cores)
        return True
    except OSError as e:
        log.warning(f"setaffinity({tid}, {cores}) failed: {e}")
        return False


def set_nice(inc):
    """Adjust the niceness of the current process.

    ``inc`` is relative to the current nice value (negative = higher
    priority, requires CAP_SYS_NICE or root).  Typical range: -20 to 19.
    """
    try:
        current = os.nice(0)
        os.nice(inc)
        new = os.nice(0)
        log.info(f"Nice: {current} → {new}")
    except PermissionError:
        log.warning("Cannot renice — root or CAP_SYS_NICE required for negative values")


def pin_pipeline_worker(cores=None):
    """Convenience: pin the calling thread to ``cores``.

    Call this at the top of FaceWorker._run() to isolate NPU work onto
    specific cores so it doesn't compete with the main loop.
    """
    if cores is None:
        return  # no-op — let scheduler decide
    pin_thread(0, cores)
