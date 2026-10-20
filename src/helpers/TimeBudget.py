"""Per-task time budget shared by every solver stage running in this process.

With ``solver.total_time_budget`` the task limit covers the whole task (PSO,
matheuristic stages, model construction and the final MIP), so each CPLEX call
receives only the time that is left. The state is process-local: a worker runs one
task at a time, and ``InstanceProcess.process`` starts and clears it.
"""

import time

_started_at = None
_budget_seconds = None


def start(budget_seconds):
    global _started_at, _budget_seconds
    _started_at = time.perf_counter()
    _budget_seconds = float(budget_seconds)


def clear():
    global _started_at, _budget_seconds
    _started_at = None
    _budget_seconds = None


def active():
    return _started_at is not None


def elapsed():
    """Seconds since ``start``, or ``None`` when no budget is active."""
    if _started_at is None:
        return None
    return time.perf_counter() - _started_at


def remaining():
    """Seconds left in the budget, or ``None`` when no budget is active."""
    if _started_at is None:
        return None
    return _budget_seconds - (time.perf_counter() - _started_at)


def clamp(limit, minimum=1.0):
    """Limit ``limit`` to the time left; unchanged when no budget is active."""
    left = remaining()
    if left is None or limit is None:
        return limit
    return min(float(limit), max(float(minimum), left))
