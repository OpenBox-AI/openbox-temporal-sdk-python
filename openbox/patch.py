"""BLOCK-with-patch request envelope, normalizer, and error-transport extractor.

Pure and workflow-sandbox-safe: at module top level this imports ONLY stdlib
(``dataclasses`` / ``typing``) and the pure base contract
``openbox_core.contracts.results``. No ``httpx`` / ``temporalio`` / ``logging`` /
crypto, so it is safe to import on a workflow-sandbox path (guarded by
``tests/test_workflow_sandbox_import_safety.py``).

Every governance response — workflow lifecycle, signal, activity lifecycle, hook
(started/completed), handoff, or HITL approval poll — passes through
:func:`patch_request` (which delegates to the base ``handle_patch``) before any
BLOCK / HALT / guardrail / HITL enforcement. No caller inspects a raw ``patch``
directly.

The Temporal restart coordinator, restart-budget helper, and the internal
control-flow exception live in the sibling sandbox-safe module
``openbox/patch_coordinator.py`` (it additionally imports ``temporalio.workflow``);
this module stays free of any ``temporalio`` dependency so the pure envelope is
independently importable and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openbox_core.contracts.results import (
    ApprovalResult,
    EvaluationResult,
    handle_patch,
)

from .errors import GOVERNANCE_PATCH_ERROR_TYPE, GOVERNANCE_RETRYABLE_BLOCK_ERROR_TYPE

__all__ = [
    "GOVERNANCE_PATCH_SCHEMA_VERSION",
    "PatchRequest",
    "patch_request",
    "extract_patch_request",
]

# Envelope schema version. Bump ONLY on a breaking change to the detail payload;
# ``from_dict`` rejects any other version (fail safe as plain BLOCK).
GOVERNANCE_PATCH_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PatchRequest:
    """A base ``PatchDirective`` wrapped with Temporal origin metadata.

    ``new_input`` follows the base wire contract: ``None`` (reuse the current run's
    input) | str | number | list | dict.

    ``event_type`` / ``hook_trigger`` / ``hook_stage`` are ORIGIN METADATA ONLY —
    never authorization. The single authorization signal is that the base
    ``handle_patch`` returned a directive.
    """

    schema_version: int
    new_input: Any
    governance_event_id: str | None
    reason: str | None
    event_type: str
    hook_trigger: bool
    hook_stage: str | None

    def to_dict(self) -> dict:
        """Serialize to the ``ApplicationError`` detail payload (every field explicit)."""
        return {
            "schema_version": self.schema_version,
            "new_input": self.new_input,
            "governance_event_id": self.governance_event_id,
            "reason": self.reason,
            "event_type": self.event_type,
            "hook_trigger": self.hook_trigger,
            "hook_stage": self.hook_stage,
        }

    @classmethod
    def from_dict(cls, data: Any) -> PatchRequest | None:
        """Reconstruct from a detail dict, or return ``None`` on ANY contract
        violation (fail safe as plain BLOCK). Never raises.

        The ``new_input`` KEY must be PRESENT even when its value is ``null``: a
        present ``None`` is the valid "reuse current input" directive, while an
        ABSENT key is a malformed envelope (mirrors the base ``_parse_patch``
        requiring exactly ``{new_input}``).
        """
        if not isinstance(data, dict):
            return None
        if data.get("schema_version") != GOVERNANCE_PATCH_SCHEMA_VERSION:
            return None
        # Present-null is the valid reuse directive; an absent key is malformed.
        if "new_input" not in data:
            return None
        event_type = data.get("event_type")
        if not isinstance(event_type, str) or not event_type:
            return None
        # bool is an int subclass — reject 0/1/other non-bool explicitly.
        hook_trigger = data.get("hook_trigger")
        if not isinstance(hook_trigger, bool):
            return None
        hook_stage = data.get("hook_stage")
        if hook_stage is not None and not isinstance(hook_stage, str):
            return None
        governance_event_id = data.get("governance_event_id")
        if governance_event_id is not None and not isinstance(governance_event_id, str):
            return None
        reason = data.get("reason")
        if reason is not None and not isinstance(reason, str):
            return None
        return cls(
            schema_version=GOVERNANCE_PATCH_SCHEMA_VERSION,
            new_input=data["new_input"],
            governance_event_id=governance_event_id,
            reason=reason,
            event_type=event_type,
            hook_trigger=hook_trigger,
            hook_stage=hook_stage,
        )


def patch_request(
    result: EvaluationResult | ApprovalResult,
    *,
    event_type: str,
    hook_trigger: bool = False,
    hook_stage: str | None = None,
) -> PatchRequest | None:
    """Wrap a base patch directive with Temporal origin metadata, or return ``None``.

    Delegates the entire verdict matrix (exact BLOCK + valid patch vs HALT / plain
    BLOCK / malformed patch / expired / pending) to the base ``handle_patch``;
    this SDK never re-implements it.
    """
    directive = handle_patch(result)
    if directive is None:
        return None
    return PatchRequest(
        schema_version=GOVERNANCE_PATCH_SCHEMA_VERSION,
        new_input=directive.new_input,
        governance_event_id=directive.governance_event_id,
        reason=directive.reason,
        event_type=event_type,
        hook_trigger=hook_trigger,
        hook_stage=hook_stage,
    )


def extract_patch_request(
    exc: BaseException,
) -> PatchRequest | None:
    """Recover a :class:`PatchRequest` from a Temporal exception chain.

    Walks ``cause`` / ``__cause__`` / ``__context__`` (mirrors
    ``workflow_interceptor._application_error_type``), matches an
    ``ApplicationError`` whose ``type`` equals EITHER the current stable
    ``GovernancePatch`` constant OR the legacy pre-rename ``GovernanceRetryableBlock``
    alias (never the human-readable message) — already-recorded histories may
    still carry the legacy type on a replayed/pending restart chain, so both
    must stay extractable. Requires EXACTLY ONE dict detail and validates it via
    ``from_dict``. Any mismatch → ``None`` (fail safe as plain BLOCK), never a
    Workflow Task retry loop.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, "type", None) in (
            GOVERNANCE_PATCH_ERROR_TYPE,
            GOVERNANCE_RETRYABLE_BLOCK_ERROR_TYPE,
        ):
            details = getattr(current, "details", None)
            if isinstance(details, (list, tuple)) and len(details) == 1:
                return PatchRequest.from_dict(details[0])
            # Right type but wrong detail shape → fail safe as plain BLOCK.
            return None
        next_exc = (
            getattr(current, "cause", None)
            or getattr(current, "__cause__", None)
            or getattr(current, "__context__", None)
        )
        current = next_exc
    return None
