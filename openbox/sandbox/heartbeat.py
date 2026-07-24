"""Context-bound, metadata-only Temporal lifecycle heartbeats."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator


@dataclass
class _Binding:
    heartbeat: Callable[[dict[str, Any]], None]
    latest: dict[str, Any]


_binding: contextvars.ContextVar[_Binding | None] = contextvars.ContextVar(
    "openbox_governed_command_heartbeat", default=None
)


class TemporalHeartbeatSink:
    """A telemetry sink that emits only safe Temporal lifecycle metadata.

    Configure the same instance on the Temporal wrapper and on the natural
    dispatcher's telemetry path or the compatibility sandbox engine. Natural
    structural configuration cannot verify dispatcher telemetry ownership.
    Context binding keeps concurrent Activities isolated.
    """

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
    ) -> Iterator[None]:
        binding = _Binding(
            heartbeat=heartbeat,
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
            heartbeat(dict(binding.latest))
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
