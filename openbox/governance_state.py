"""Temporal-owned governance state — the small amount of Temporal semantics that
must survive PAST a base-SDK hook callback.

The base SDK (``openbox_core``) owns hook context, hook payload building, hook
evaluation, and within-activity abort short-circuit (its ``ContextStore``). This
object holds ONLY the Temporal effects the base runtime cannot express itself:

- **signal verdicts** — a SignalReceived BLOCK/HALT must fail the *next* activity
  in the same run (workflow interceptor records; activity interceptor enforces).
- **HITL pending-approval markers** — a REQUIRE_APPROVAL raises a retryable error;
  the next attempt must POLL approval status instead of re-evaluating.
- **completed-hook stop bridge** — a completed-hook BLOCK/HALT resolved by the
  base runtime is recorded here (keyed by workflow/run/activity) so the activity
  interceptor can skip a duplicate completed event (BLOCK) or reach Temporal's
  terminate path (HALT) after user code returns, then clear it.

All keys are RUN-SCOPED: state from a prior run with the same ``workflow_id`` is
ignored and cleared. Thread-safe — activities run on worker threads.
"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

from .types import Verdict

__all__ = ["TemporalGovernanceState"]

# (workflow_id, run_id, activity_id)
_RunActivityKey = Tuple[str, str, str]


class TemporalGovernanceState:
    """Run-scoped Temporal governance effects, shared across interceptors."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # workflow_id -> (verdict, reason, run_id)
        self._signal_verdicts: dict[str, Tuple[Verdict, Optional[str], str]] = {}
        # (workflow_id, run_id, activity_id) awaiting a HITL decision
        self._pending_approval: set[_RunActivityKey] = set()
        # (workflow_id, run_id, activity_id) -> (verdict, reason) from a completed hook
        self._completed_stop: dict[_RunActivityKey, Tuple[Verdict, Optional[str]]] = {}

    def set_signal_verdict(
        self, workflow_id: str, run_id: str, verdict: Verdict, reason: Optional[str] = None
    ) -> None:
        """Record a SignalReceived BLOCK/HALT that must fail the next activity."""
        with self._lock:
            self._signal_verdicts[workflow_id] = (verdict, reason, run_id)

    def get_signal_verdict(
        self, workflow_id: str, run_id: str
    ) -> Optional[Tuple[Verdict, Optional[str]]]:
        """Peek the pending signal verdict for this run. Clears (and ignores) a
        verdict left by a prior run with the same workflow_id."""
        with self._lock:
            entry = self._signal_verdicts.get(workflow_id)
            if entry is None:
                return None
            verdict, reason, stored_run = entry
            if stored_run != run_id:
                del self._signal_verdicts[workflow_id]
                return None
            return verdict, reason

    def mark_pending_approval(self, workflow_id: str, run_id: str, activity_id: str) -> None:
        with self._lock:
            self._pending_approval.add((workflow_id, run_id, activity_id))

    def has_pending_approval(self, workflow_id: str, run_id: str, activity_id: str) -> bool:
        with self._lock:
            return (workflow_id, run_id, activity_id) in self._pending_approval

    def clear_pending_approval(self, workflow_id: str, run_id: str, activity_id: str) -> None:
        with self._lock:
            self._pending_approval.discard((workflow_id, run_id, activity_id))

    def record_completed_stop(
        self,
        workflow_id: str,
        run_id: str,
        activity_id: str,
        verdict: Verdict,
        reason: Optional[str] = None,
    ) -> None:
        """A completed-hook BLOCK/HALT resolved by the base runtime. Affects only
        FUTURE execution — the operation already ran."""
        with self._lock:
            self._completed_stop[(workflow_id, run_id, activity_id)] = (verdict, reason)

    def take_completed_stop(
        self, workflow_id: str, run_id: str, activity_id: str
    ) -> Optional[Tuple[Verdict, Optional[str]]]:
        """Return AND clear the completed-hook stop for this exact run/activity.

        Consumed on EVERY activity exit path (success in _handle_completion,
        failure in _consume_completed_halt), so entries never strand."""
        with self._lock:
            return self._completed_stop.pop((workflow_id, run_id, activity_id), None)
