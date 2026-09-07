"""Context-bound, metadata-only Temporal lifecycle heartbeats."""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass
class _Binding:
    heartbeat: Callable[[dict[str, Any]], None]
    latest: dict[str, Any]
    telemetry_owner: Any = None


_binding: contextvars.ContextVar[_Binding | None] = contextvars.ContextVar(
    "openbox_governed_command_heartbeat", default=None
)


class TemporalHeartbeatSink:
    """A dispatcher telemetry sink that emits only safe lifecycle metadata.

    The same instance must be configured on the standalone dispatcher and the
    Temporal adapter. Context binding keeps concurrent Activities isolated.
    """

    def __init__(self) -> None:
        self._otel_bridge: Any = None

    def attach_otel_bridge(self, bridge: Any) -> None:
        """Attach the optional v3 collector without changing sink identity."""
        if self._otel_bridge is not None:
            raise ValueError("governed-command telemetry bridge already attached")
        self._otel_bridge = bridge

    @contextmanager
    def bind(
        self,
        heartbeat: Callable[[dict[str, Any]], None],
        *,
        workflow_id: str,
        run_id: str,
        activity_id: str,
        attempt: int,
        profile_id: str,
        telemetry_owner: Any = None,
    ) -> Iterator[None]:
        binding = _Binding(
            heartbeat=heartbeat,
            telemetry_owner=telemetry_owner,
            latest={
                "phase": "governed_dispatch_started",
                "workflow_id": workflow_id,
                "run_id": run_id,
                "activity_id": activity_id,
                "attempt": attempt,
                "profile_id": profile_id,
            },
        )
        token = _binding.set(binding)
        try:
            # Heartbeat first: cancellation and every heartbeat BaseException
            # must propagate unchanged before optional lifecycle collection.
            heartbeat(dict(binding.latest))
            if self._otel_bridge is not None:
                try:
                    self._otel_bridge.record_started(telemetry_owner)
                except Exception:
                    pass
            yield
        finally:
            _binding.reset(token)

    async def emit(self, event: Any) -> None:
        binding = _binding.get()
        if binding is None:
            return
        binding.latest = {
            "phase": event.lifecycle_phase or event.event,
            "event": event.event,
            "workflow_id": event.workflow_id,
            "run_id": event.run_id,
            "activity_id": event.activity_id,
            "attempt": event.attempt,
            "profile_id": binding.latest["profile_id"],
            "disposition": event.disposition,
            "sandbox_id": event.sandbox_id,
        }
        binding.heartbeat(
            {key: value for key, value in binding.latest.items() if value is not None}
        )
        if self._otel_bridge is not None:
            try:
                self._otel_bridge.record_phase(binding.telemetry_owner, event)
            except Exception:
                pass

    def heartbeat_latest(self) -> None:
        binding = _binding.get()
        if binding is not None:
            binding.heartbeat(
                {
                    key: value
                    for key, value in binding.latest.items()
                    if value is not None
                }
            )
