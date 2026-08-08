"""Bounded, provider-owned OpenTelemetry bridge for governed commands.

The Activity path only captures the current ``SpanContext`` and immutable facts,
then performs a non-blocking queue write.  A single daemon thread owns every
interaction with the application's already-configured global providers.
"""

from __future__ import annotations

import asyncio
import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping, cast

QUEUE_CAPACITY = 1024
MAX_PHASE_FACTS = 64
SHUTDOWN_TIMEOUT_SECONDS = 5.0
_MAX_COUNTER = 2**63 - 1
_IMAGE = re.compile(r"[^\s]+@(sha256:[0-9a-f]{64})\Z")
_SAFE_TEXT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*\Z")

_EVENT_NAMES = frozenset(
    {
        "governed_dispatch_started",
        "governance_decision_received",
        "host_exec_finished",
        "sandbox_create_started",
        "sandbox_create_finished",
        "sandbox_ready",
        "sandbox_exec_started",
        "sandbox_exec_finished",
        "sandbox_delete_started",
        "sandbox_deleted",
        "sandbox_execution_failed",
        "dispatch_terminal",
    }
)
_PHASES = frozenset({"dispatch", "governance", "host", "create", "ready", "exec", "delete"})
_DISPOSITIONS = frozenset(
    {
        "unknown",
        "executed_on_host",
        "executed_in_sandbox",
        "not_executed",
        "execution_indeterminate",
    }
)
_TIMEOUT_STATUSES = frozenset({"unknown", "not_observed", "confirmed_timeout", "possible_timeout"})
_CLEANUP_STATUSES = frozenset({"unknown", "not_needed", "deleted", "failed"})
_DIRECTIVES = frozenset({"unknown", "continue", "halt"})
_ERROR_CODES = frozenset(
    {
        "unknown",
        "none",
        "invalid_command",
        "profile_rejected",
        "governance_transport",
        "governance_protocol",
        "governance_fallback",
        "unsupported_constraint",
        "remediation_unsupported",
        "approval_required",
        "blocked",
        "halted",
        "sandbox_disabled",
        "sandbox_create_failed",
        "sandbox_readiness_failed",
        "sandbox_exec_not_dispatched",
        "sandbox_exec_indeterminate",
        "sandbox_protocol_failed",
        "host_exec_indeterminate",
        "host_output_limit",
        "cancelled",
    }
)
_OUTCOMES = frozenset(
    {
        "success",
        "nonzero",
        "timeout",
        "not_executed",
        "dispatcher_error",
        "cancelled",
        "base_exception",
    }
)


def parse_image_digest(template: str) -> str:
    """Return only a lowercase digest from one full immutable image reference."""
    if not isinstance(template, str):
        raise ValueError("immutable image template rejected")
    matched = _IMAGE.fullmatch(template)
    if matched is None:
        raise ValueError("immutable image template rejected")
    return matched.group(1)


def _safe_text(value: object, maximum: int = 256) -> str:
    if (
        not isinstance(value, str)
        or len(value.encode("utf-8")) > maximum
        or _SAFE_TEXT.fullmatch(value) is None
    ):
        return "unknown"
    return value


def _finite(value: object, allowed: frozenset[str]) -> str:
    return value if isinstance(value, str) and value in allowed else "unknown"


def _bounded_count(value: object) -> int:
    return value if type(value) is int and 0 <= value <= 2**31 - 1 else 0


@dataclass(frozen=True, slots=True)
class GovernedCommandPhaseFact:
    """One lifecycle observation with optional explicitly unsafe raw fields."""

    timestamp_ns: int
    event: str
    phase: str
    disposition: str
    timeout_status: str
    cleanup_status: str
    raw_attributes: tuple[tuple[str, str | int | float | bool], ...] = ()


@dataclass(frozen=True, slots=True)
class GovernedCommandTerminalRecord:
    """Immutable terminal snapshot handed from an Activity to the daemon."""

    workflow_id: str
    run_id: str
    activity_id: str
    attempt: int
    profile_id: str
    workflow_type: str
    task_queue: str
    image_digest: str
    sandbox_provider: str | None
    parent_span_context: Any
    started_ns: int
    ended_ns: int
    phases: tuple[GovernedCommandPhaseFact, ...]
    outcome: str
    disposition: str
    timeout_status: str
    cleanup_status: str
    directive: str
    error_code: str
    exit_code: int | None
    stdout_bytes: int
    stderr_bytes: int
    unsafe_stdout: str | None
    unsafe_stderr: str | None


class GovernedCommandTelemetryOwner:
    """Mutable request-local owner which can produce one terminal snapshot."""

    __slots__ = (
        "activity_id",
        "attempt",
        "parent_span_context",
        "phases",
        "profile_id",
        "run_id",
        "started_ns",
        "task_queue",
        "workflow_id",
        "workflow_type",
        "_finalized",
        "_key",
        "_lock",
    )

    def __init__(
        self,
        *,
        workflow_id: str,
        run_id: str,
        activity_id: str,
        attempt: int,
        profile_id: str,
        workflow_type: str,
        task_queue: str,
        parent_span_context: Any,
        owner_key: tuple[str, str, str, int],
    ) -> None:
        self.workflow_id = _safe_text(workflow_id)
        self.run_id = _safe_text(run_id)
        self.activity_id = _safe_text(activity_id)
        self.attempt = attempt if type(attempt) is int and 1 <= attempt <= 2**31 - 1 else 1
        self.profile_id = _safe_text(profile_id, 128)
        self.workflow_type = _safe_text(workflow_type)
        self.task_queue = _safe_text(task_queue)
        self.parent_span_context = parent_span_context
        self.started_ns = time.time_ns()
        self._key = owner_key
        self.phases: list[GovernedCommandPhaseFact] = []
        self._finalized = False
        self._lock = threading.Lock()


class GovernedCommandTelemetryBridge:
    """Non-blocking Activity adapter backed by one bounded provider daemon."""

    def __init__(
        self,
        image_template: str,
        *,
        sandbox_provider: str | None = None,
        include_unsafe_attributes: bool = False,
        _shutdown_timeout_seconds: float = SHUTDOWN_TIMEOUT_SECONDS,
    ) -> None:
        self.image_digest = parse_image_digest(image_template)
        if sandbox_provider is not None and _safe_text(sandbox_provider, 128) == "unknown":
            raise ValueError("sandbox provider metadata rejected")
        self.sandbox_provider = sandbox_provider
        self.include_unsafe_attributes = include_unsafe_attributes
        self._queue: queue.Queue[GovernedCommandTerminalRecord] = queue.Queue(
            maxsize=QUEUE_CAPACITY
        )
        self._shutdown_timeout_seconds = _shutdown_timeout_seconds
        self._state_lock = threading.Lock()
        self._drop_lock = threading.Lock()
        self._owners: set[tuple[str, str, str, int]] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._accepting = True
        self._span_records_dropped = 0
        self._metric_batches_dropped = 0

    @property
    def drop_counts(self) -> tuple[int, int]:
        with self._drop_lock:
            return self._span_records_dropped, self._metric_batches_dropped

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def worker_thread(self) -> threading.Thread | None:
        return self._thread

    def start(self) -> None:
        """Start the sole daemon without touching an OTel provider."""
        with self._state_lock:
            if self._thread is not None or not self._accepting:
                return
            self._thread = threading.Thread(
                target=self._run,
                name="openbox-governed-command-otel",
                daemon=True,
            )
            self._thread.start()

    def shutdown(self) -> None:
        """Stop accepting records and wait no longer than the fixed bound."""
        with self._state_lock:
            self._accepting = False
            thread = self._thread
            self._stop.set()
        if thread is not None:
            thread.join(timeout=self._shutdown_timeout_seconds)

    def begin(
        self,
        *,
        workflow_id: str,
        run_id: str,
        activity_id: str,
        attempt: int,
        profile_id: str,
        workflow_type: str,
        task_queue: str,
    ) -> GovernedCommandTelemetryOwner | None:
        """Capture only the explicit current span context and claim one owner."""
        key = (workflow_id, run_id, activity_id, attempt)
        with self._state_lock:
            if not self._accepting or key in self._owners or len(self._owners) >= QUEUE_CAPACITY:
                return None
            self._owners.add(key)
        try:
            from opentelemetry.trace import (
                SpanContext,
                TraceFlags,
                TraceState,
                get_current_span,
            )

            current = get_current_span().get_span_context()
            parent = (
                SpanContext(
                    trace_id=int(current.trace_id),
                    span_id=int(current.span_id),
                    is_remote=False,
                    trace_flags=TraceFlags(int(current.trace_flags)),
                    trace_state=TraceState(),
                )
                if current.is_valid
                else None
            )
        except Exception:
            parent = None
        return GovernedCommandTelemetryOwner(
            workflow_id=workflow_id,
            run_id=run_id,
            activity_id=activity_id,
            attempt=attempt,
            profile_id=profile_id,
            workflow_type=workflow_type,
            task_queue=task_queue,
            parent_span_context=parent,
            owner_key=key,
        )

    def record_started(self, owner: GovernedCommandTelemetryOwner | None) -> None:
        self._append_phase(owner, "governed_dispatch_started", "dispatch", None)

    def record_phase(self, owner: GovernedCommandTelemetryOwner | None, event: object) -> None:
        """Snapshot only allowlisted fields from a dispatcher lifecycle event."""
        if owner is None:
            return
        try:
            if (
                getattr(event, "workflow_id", None) != owner.workflow_id
                or getattr(event, "run_id", None) != owner.run_id
                or getattr(event, "activity_id", None) != owner.activity_id
                or getattr(event, "attempt", None) != owner.attempt
            ):
                return
            name = getattr(event, "event", None)
            phase = getattr(event, "lifecycle_phase", None)
            if name == "governance_decision_received":
                phase = "governance"
            elif name == "host_exec_finished":
                phase = "host"
            elif name == "dispatch_terminal":
                phase = "dispatch"
            self._append_phase(owner, name, phase, event)
        except Exception:
            return

    def _append_phase(
        self,
        owner: GovernedCommandTelemetryOwner | None,
        name: object,
        phase: object,
        event: object | None,
    ) -> None:
        if owner is None or name not in _EVENT_NAMES or phase not in _PHASES:
            return
        raw_attributes: tuple[tuple[str, str | int | float | bool], ...] = ()
        if self.include_unsafe_attributes and event is not None:
            to_wire = getattr(event, "to_wire", None)
            raw = to_wire() if callable(to_wire) else vars(event)
            if isinstance(raw, dict):
                raw_attributes = tuple(
                    (str(key), value)
                    for key, value in raw.items()
                    if isinstance(value, (str, int, float, bool))
                )
        fact = GovernedCommandPhaseFact(
            timestamp_ns=time.time_ns(),
            event=cast(str, name),
            phase=cast(str, phase),
            disposition=_finite(
                None if event is None else getattr(event, "disposition", None),
                _DISPOSITIONS,
            ),
            timeout_status=_finite(
                None if event is None else getattr(event, "timeout_status", None),
                _TIMEOUT_STATUSES,
            ),
            cleanup_status=_finite(
                None if event is None else getattr(event, "cleanup_status", None),
                _CLEANUP_STATUSES,
            ),
            raw_attributes=raw_attributes,
        )
        with owner._lock:
            if not owner._finalized and len(owner.phases) < MAX_PHASE_FACTS:
                owner.phases.append(fact)

    def finalize(
        self,
        owner: GovernedCommandTelemetryOwner | None,
        *,
        dispatch_result: object | None,
        error: BaseException | None,
    ) -> GovernedCommandTerminalRecord | None:
        """Freeze and enqueue exactly one terminal record without blocking."""
        if owner is None:
            return None
        (
            outcome,
            disposition,
            timeout_status,
            cleanup_status,
            directive,
            error_code,
            exit_code,
            stdout_bytes,
            stderr_bytes,
        ) = self._terminal_metadata(dispatch_result, error)
        unsafe_stdout: str | None = None
        unsafe_stderr: str | None = None
        if self.include_unsafe_attributes and dispatch_result is not None:
            execution = getattr(dispatch_result, "execution", None)
            stdout = None if execution is None else getattr(execution, "stdout", None)
            stderr = None if execution is None else getattr(execution, "stderr", None)
            if isinstance(stdout, bytes):
                unsafe_stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                unsafe_stderr = stderr.decode("utf-8", errors="replace")
        with owner._lock:
            if owner._finalized:
                return None
            owner._finalized = True
            record = GovernedCommandTerminalRecord(
                workflow_id=owner.workflow_id,
                run_id=owner.run_id,
                activity_id=owner.activity_id,
                attempt=owner.attempt,
                profile_id=owner.profile_id,
                workflow_type=owner.workflow_type,
                task_queue=owner.task_queue,
                image_digest=self.image_digest,
                sandbox_provider=self.sandbox_provider,
                parent_span_context=owner.parent_span_context,
                started_ns=owner.started_ns,
                ended_ns=max(owner.started_ns, time.time_ns()),
                phases=tuple(owner.phases),
                outcome=outcome,
                disposition=disposition,
                timeout_status=timeout_status,
                cleanup_status=cleanup_status,
                directive=directive,
                error_code=error_code,
                exit_code=exit_code,
                stdout_bytes=stdout_bytes,
                stderr_bytes=stderr_bytes,
                unsafe_stdout=unsafe_stdout,
                unsafe_stderr=unsafe_stderr,
            )
        with self._state_lock:
            self._owners.discard(owner._key)
            accepting = self._accepting
        if not accepting:
            return record
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            with self._drop_lock:
                self._span_records_dropped = min(_MAX_COUNTER, self._span_records_dropped + 1)
                self._metric_batches_dropped = min(_MAX_COUNTER, self._metric_batches_dropped + 1)
        return record

    @staticmethod
    def _terminal_metadata(
        result: object | None, error: BaseException | None
    ) -> tuple[str, str, str, str, str, str, int | None, int, int]:
        if isinstance(error, asyncio.CancelledError):
            return (
                "cancelled",
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                "cancelled",
                None,
                0,
                0,
            )
        if error is not None:
            # Cancellation is a real outcome, not a dispatcher error — drain paths
            # propagate asyncio.CancelledError through the dispatch task without
            # producing a DispatchResult. Report as "cancelled" so downstream OTel
            # invariants line up with the interceptor's own "cancelled" completion.
            if isinstance(error, asyncio.CancelledError):
                return (
                    "cancelled",
                    "unknown",
                    "unknown",
                    "unknown",
                    "unknown",
                    "cancelled",
                    None,
                    0,
                    0,
                )
            return (
                ("dispatcher_error" if isinstance(error, Exception) else "base_exception"),
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                None,
                0,
                0,
            )
        try:
            disposition = _finite(getattr(getattr(result, "disposition"), "value"), _DISPOSITIONS)
            directive = _finite(getattr(getattr(result, "directive"), "value"), _DIRECTIVES)
            dispatch_error = getattr(result, "error", None)
            error_code = (
                "none"
                if dispatch_error is None
                else _finite(getattr(getattr(dispatch_error, "code"), "value"), _ERROR_CODES)
            )
            execution = getattr(result, "execution", None)
            timeout_status = _finite(
                (
                    None
                    if execution is None
                    else getattr(getattr(execution, "timeout_status"), "value")
                ),
                _TIMEOUT_STATUSES,
            )
            cleanup_status = _finite(
                (
                    None
                    if execution is None
                    else getattr(getattr(execution, "cleanup_status"), "value")
                ),
                _CLEANUP_STATUSES,
            )
            raw_exit_code = None if execution is None else getattr(execution, "exit_code", None)
            exit_code = (
                raw_exit_code
                if type(raw_exit_code) is int and -(2**31) <= raw_exit_code < 2**31
                else None
            )
            stdout = None if execution is None else getattr(execution, "stdout", None)
            stderr = None if execution is None else getattr(execution, "stderr", None)
            stdout_bytes = _bounded_count(len(stdout) if isinstance(stdout, bytes) else None)
            stderr_bytes = _bounded_count(len(stderr) if isinstance(stderr, bytes) else None)
        except Exception:
            return (
                "dispatcher_error",
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                None,
                0,
                0,
            )
        if error_code == "cancelled":
            outcome = "cancelled"
        elif timeout_status in {"confirmed_timeout", "possible_timeout"}:
            outcome = "timeout"
        elif disposition in {"executed_on_host", "executed_in_sandbox"}:
            outcome = "success" if exit_code == 0 else "nonzero"
        else:
            outcome = "not_executed"
        # `outcome=="cancelled"` takes precedence: even if a partial sandbox exec
        # bumped exit_code, the cancel is the terminal outcome the dispatcher
        # signalled to Core; the Worker restart proof asserts this exact value.
        return (
            outcome,
            disposition,
            timeout_status,
            cleanup_status,
            directive,
            error_code,
            exit_code,
            stdout_bytes,
            stderr_bytes,
        )

    def _run(self) -> None:
        tracer = None
        meter = None
        completion_counter = None
        duration_histogram = None
        dropped_counter = None
        tracer_provider = None
        meter_provider = None
        try:
            from opentelemetry import trace

            tracer_provider = trace.get_tracer_provider()
            tracer = tracer_provider.get_tracer("openbox.governed_commands", "3")
        except Exception:
            tracer = None
        try:
            from opentelemetry import metrics

            meter_provider = metrics.get_meter_provider()
            meter = meter_provider.get_meter("openbox.governed_commands", "3")
        except Exception:
            meter = None
        if meter is not None:
            try:
                completion_counter = meter.create_counter("openbox.governed_command.completions")
            except Exception:
                completion_counter = None
            try:
                duration_histogram = meter.create_histogram(
                    "openbox.governed_command.duration", unit="ms"
                )
            except Exception:
                duration_histogram = None
            try:
                dropped_counter = meter.create_counter("openbox.governed_command.telemetry.dropped")
            except Exception:
                dropped_counter = None

        reported_span_drops = 0
        reported_metric_drops = 0
        while not self._stop.is_set():
            try:
                record = self._queue.get(timeout=0.05)
            except queue.Empty:
                reported_span_drops, reported_metric_drops = self._report_drops(
                    dropped_counter, reported_span_drops, reported_metric_drops
                )
                continue
            if tracer is not None:
                self._record_span(tracer, record)
            self._record_metrics(completion_counter, duration_histogram, record)
            reported_span_drops, reported_metric_drops = self._report_drops(
                dropped_counter, reported_span_drops, reported_metric_drops
            )
        self._report_drops(dropped_counter, reported_span_drops, reported_metric_drops)
        for provider in (tracer_provider, meter_provider):
            try:
                flush = getattr(provider, "force_flush", None)
                if callable(flush):
                    flush()
            except Exception:
                pass

    @staticmethod
    def _record_span(tracer: object, record: GovernedCommandTerminalRecord) -> None:
        try:
            from opentelemetry.context import Context
            from opentelemetry.trace import (
                NonRecordingSpan,
                SpanKind,
                Status,
                StatusCode,
                set_span_in_context,
            )

            parent = Context()
            span_context = record.parent_span_context
            if span_context is not None and getattr(span_context, "is_valid", False):
                parent = set_span_in_context(NonRecordingSpan(span_context), parent)
            attributes: dict[str, str | int] = {
                "openbox.governed.workflow_id": record.workflow_id,
                "openbox.governed.run_id": record.run_id,
                "openbox.governed.activity_id": record.activity_id,
                "openbox.governed.attempt": record.attempt,
                "openbox.governed.profile_id": record.profile_id,
                "openbox.governed.workflow_type": record.workflow_type,
                "openbox.governed.task_queue": record.task_queue,
                "openbox.governed.image_digest": record.image_digest,
            }
            # `openbox.hook.type` identifies the OTel event category, not the
            # dispatcher's disposition; downstream validators require it to be
            # present on every governed_command terminal span, including
            # cancelled/host paths where disposition may be "unknown".
            attributes["openbox.hook.type"] = "sandbox_execution"
            if (
                record.disposition == "executed_in_sandbox"
                and record.sandbox_provider is not None
            ):
                attributes["sandbox.provider"] = record.sandbox_provider
            span = tracer.start_span(  # type: ignore[attr-defined]
                "openbox.governed_command",
                context=parent,
                kind=SpanKind.INTERNAL,
                start_time=record.started_ns,
                attributes=attributes,
            )
        except Exception:
            return
        try:
            for fact in record.phases:
                try:
                    event_attributes: dict[str, str | int | float | bool] = {
                        "phase": fact.phase,
                        "disposition": fact.disposition,
                        "timeout_status": fact.timeout_status,
                        "cleanup_status": fact.cleanup_status,
                    }
                    event_attributes.update(
                        {"openbox.unsafe." + key: value for key, value in fact.raw_attributes}
                    )
                    span.add_event(
                        "openbox.governed_command." + fact.event,
                        attributes=event_attributes,
                        timestamp=fact.timestamp_ns,
                    )
                except Exception:
                    pass
            try:
                span.set_attributes(
                    {
                        "openbox.governed.outcome": record.outcome,
                        "openbox.governed.disposition": record.disposition,
                        "openbox.governed.timeout_status": record.timeout_status,
                        "openbox.governed.cleanup_status": record.cleanup_status,
                        "openbox.governed.directive": record.directive,
                        "openbox.governed.error_code": record.error_code,
                        "openbox.governed.stdout_bytes": record.stdout_bytes,
                        "openbox.governed.stderr_bytes": record.stderr_bytes,
                    }
                )
                if record.exit_code is not None:
                    span.set_attribute("openbox.governed.exit_code", record.exit_code)
                if record.unsafe_stdout is not None:
                    span.set_attribute("openbox.unsafe.stdout_body", record.unsafe_stdout)
                if record.unsafe_stderr is not None:
                    span.set_attribute("openbox.unsafe.stderr_body", record.unsafe_stderr)
                span.set_status(
                    Status(StatusCode.UNSET if record.outcome == "success" else StatusCode.ERROR)
                )
            except Exception:
                pass
        finally:
            try:
                span.end(end_time=record.ended_ns)
            except Exception:
                pass

    @staticmethod
    def _record_metrics(
        completion_counter: object | None,
        duration_histogram: object | None,
        record: GovernedCommandTerminalRecord,
    ) -> None:
        labels: Mapping[str, str] = {
            "outcome": record.outcome,
            "disposition": record.disposition,
        }
        if completion_counter is not None:
            try:
                completion_counter.add(1, labels)  # type: ignore[attr-defined]
            except Exception:
                pass
        if duration_histogram is not None:
            try:
                cast(Any, duration_histogram).record(
                    max(0.0, (record.ended_ns - record.started_ns) / 1_000_000),
                    labels,
                )  # type: ignore[attr-defined]
            except Exception:
                pass

    def _report_drops(
        self, counter: object | None, span_reported: int, metric_reported: int
    ) -> tuple[int, int]:
        with self._drop_lock:
            span_total = self._span_records_dropped
            metric_total = self._metric_batches_dropped
        if counter is not None:
            for total, reported, signal in (
                (span_total, span_reported, "span_record"),
                (metric_total, metric_reported, "metric_batch"),
            ):
                delta = total - reported
                if delta > 0:
                    try:
                        counter.add(  # type: ignore[attr-defined]
                            delta, {"signal": signal, "reason": "queue_full"}
                        )
                    except Exception:
                        pass
        return span_total, metric_total
