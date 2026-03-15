# openbox/span_processor.py
"""
OpenTelemetry SpanProcessor for workflow governance.

WorkflowSpanProcessor manages activity context, trace mappings, and governance
state (verdicts, abort/halt flags) for hook-level governance. Forwards spans
to fallback exporters (Jaeger, OTLP, etc.) without buffering.
"""

from typing import TYPE_CHECKING, Dict, Optional
import threading
import logging

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

from .types import WorkflowSpanBuffer, Verdict


class WorkflowSpanProcessor:
    """
    SpanProcessor that manages governance state and forwards spans to exporters.

    Responsibilities:
    - Activity context storage (for hook-level governance payload building)
    - Trace → workflow/activity ID resolution (for hook → activity linkage)
    - Workflow buffer management (verdicts, approvals, abort/halt flags)
    - Span forwarding to fallback exporter (Jaeger, OTLP, etc.)

    Thread-safe via _lock for all shared state.
    """

    def __init__(
        self,
        fallback_processor: Optional["SpanProcessor"] = None,
        ignored_url_prefixes: Optional[list] = None,
    ):
        self.fallback = fallback_processor
        self._ignored_url_prefixes = set(ignored_url_prefixes or [])
        self._buffers: Dict[str, WorkflowSpanBuffer] = {}  # workflow_id -> buffer
        self._trace_to_workflow: Dict[int, str] = {}  # trace_id (int) -> workflow_id
        self._trace_to_activity: Dict[int, str] = {}  # trace_id (int) -> activity_id
        self._verdicts: Dict[str, dict] = {}  # workflow_id -> {"verdict": Verdict, "reason": str}
        self._activity_context: Dict[str, dict] = {}  # "{workflow_id}:{activity_id}" -> event data
        self._aborted_activities: Dict[str, str] = {}  # "{workflow_id}:{activity_id}" -> abort reason
        self._halt_requests: Dict[str, str] = {}  # "{workflow_id}:{activity_id}" -> halt reason
        self._lock = threading.Lock()

    def _should_ignore_span(self, span: "ReadableSpan") -> bool:
        """Check if span should be ignored based on URL."""
        if not self._ignored_url_prefixes:
            return False
        url = span.attributes.get("http.url") if span.attributes else None
        if url:
            for prefix in self._ignored_url_prefixes:
                if url.startswith(prefix):
                    return True
        return False

    # ═══════════════════════════════════════════════════════════════════════════
    # Workflow Buffer Management
    # ═══════════════════════════════════════════════════════════════════════════

    def register_workflow(self, workflow_id: str, buffer: WorkflowSpanBuffer) -> None:
        """Register buffer for a workflow."""
        with self._lock:
            self._buffers[workflow_id] = buffer

    def register_trace(self, trace_id: int, workflow_id: str, activity_id: str = None) -> None:
        """Register trace_id → workflow_id (and activity_id) mapping for hook lookups."""
        with self._lock:
            self._trace_to_workflow[trace_id] = workflow_id
            if activity_id:
                self._trace_to_activity[trace_id] = activity_id

    def get_buffer(self, workflow_id: str) -> Optional[WorkflowSpanBuffer]:
        """Retrieve buffer without removing it."""
        with self._lock:
            return self._buffers.get(workflow_id)

    def remove_buffer(self, workflow_id: str) -> Optional[WorkflowSpanBuffer]:
        """Remove and return buffer."""
        with self._lock:
            return self._buffers.pop(workflow_id, None)

    def unregister_workflow(self, workflow_id: str) -> None:
        """Clean all state associated with a workflow to prevent memory leaks."""
        with self._lock:
            self._buffers.pop(workflow_id, None)
            self._verdicts.pop(workflow_id, None)
            for store in (self._aborted_activities, self._halt_requests, self._activity_context):
                stale = [k for k in store if k.startswith(f"{workflow_id}:")]
                for k in stale:
                    del store[k]
            stale_traces = [t for t, w in self._trace_to_workflow.items() if w == workflow_id]
            for t in stale_traces:
                del self._trace_to_workflow[t]
                self._trace_to_activity.pop(t, None)

    # ═══════════════════════════════════════════════════════════════════════════
    # Verdict Storage (workflow interceptor → activity interceptor)
    # ═══════════════════════════════════════════════════════════════════════════

    def set_verdict(self, workflow_id: str, verdict: Verdict, reason: str = None, run_id: str = None) -> None:
        """Store governance verdict for a workflow. Called when SignalReceived returns BLOCK/HALT."""
        with self._lock:
            self._verdicts[workflow_id] = {"verdict": verdict, "reason": reason, "run_id": run_id}
            if workflow_id in self._buffers:
                self._buffers[workflow_id].verdict = verdict
                self._buffers[workflow_id].verdict_reason = reason

    def get_verdict(self, workflow_id: str) -> Optional[dict]:
        """Get stored verdict for a workflow."""
        with self._lock:
            return self._verdicts.get(workflow_id)

    def clear_verdict(self, workflow_id: str) -> None:
        """Clear stored verdict for a workflow."""
        with self._lock:
            self._verdicts.pop(workflow_id, None)

    # ═══════════════════════════════════════════════════════════════════════════
    # Activity Context Storage (for hook-level governance)
    # ═══════════════════════════════════════════════════════════════════════════

    def set_activity_context(self, workflow_id: str, activity_id: str, context: dict) -> None:
        """Store ActivityStarted event data for hook-level governance payload building."""
        with self._lock:
            self._activity_context[f"{workflow_id}:{activity_id}"] = context

    def get_activity_context_by_trace(self, trace_id: int) -> Optional[dict]:
        """Look up activity context using trace_id from a child span (hook → activity linkage)."""
        with self._lock:
            workflow_id = self._trace_to_workflow.get(trace_id)
            activity_id = self._trace_to_activity.get(trace_id)
            if workflow_id and activity_id:
                return self._activity_context.get(f"{workflow_id}:{activity_id}")
            return None

    def clear_activity_context(self, workflow_id: str, activity_id: str) -> None:
        """Clear buffered activity context after activity completes."""
        with self._lock:
            self._activity_context.pop(f"{workflow_id}:{activity_id}", None)

    # ═══════════════════════════════════════════════════════════════════════════
    # Activity Abort Signal (block subsequent hooks after BLOCK/HALT/REQUIRE_APPROVAL)
    # ═══════════════════════════════════════════════════════════════════════════

    def set_activity_abort(self, workflow_id: str, activity_id: str, reason: str) -> None:
        """Set abort flag for an activity. Subsequent hooks will raise immediately."""
        with self._lock:
            self._aborted_activities[f"{workflow_id}:{activity_id}"] = reason

    def get_activity_abort(self, workflow_id: str, activity_id: str) -> Optional[str]:
        """Check if activity is aborted. Returns reason string or None."""
        with self._lock:
            return self._aborted_activities.get(f"{workflow_id}:{activity_id}")

    def clear_activity_abort(self, workflow_id: str, activity_id: str) -> None:
        """Clear abort flag for an activity (on retry or completion)."""
        with self._lock:
            self._aborted_activities.pop(f"{workflow_id}:{activity_id}", None)

    # ═══════════════════════════════════════════════════════════════════════════
    # Halt Request (hook → activity interceptor for HALT verdict)
    # ═══════════════════════════════════════════════════════════════════════════

    def set_halt_requested(self, workflow_id: str, activity_id: str, reason: str) -> None:
        """Hook sets this when HALT verdict received. Activity interceptor calls terminate()."""
        with self._lock:
            self._halt_requests[f"{workflow_id}:{activity_id}"] = reason

    def get_halt_requested(self, workflow_id: str, activity_id: str) -> Optional[str]:
        """Check if HALT was requested by a hook. Returns reason or None."""
        with self._lock:
            return self._halt_requests.get(f"{workflow_id}:{activity_id}")

    def clear_halt_requested(self, workflow_id: str, activity_id: str) -> None:
        """Clear halt request flag."""
        with self._lock:
            self._halt_requests.pop(f"{workflow_id}:{activity_id}", None)

    # ═══════════════════════════════════════════════════════════════════════════
    # SpanProcessor Interface
    # ═══════════════════════════════════════════════════════════════════════════

    def on_start(self, span, parent_context=None) -> None:
        """Called when span starts. No-op."""
        pass

    def on_end(self, span: "ReadableSpan") -> None:
        """Called when span ends. Forward to fallback exporter only."""
        if self._should_ignore_span(span):
            if self.fallback:
                self.fallback.on_end(span)
            return

        if self.fallback:
            self.fallback.on_end(span)

    def shutdown(self) -> None:
        """Shutdown the processor."""
        if self.fallback:
            self.fallback.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any buffered spans."""
        if self.fallback:
            return self.fallback.force_flush(timeout_millis)
        return True
