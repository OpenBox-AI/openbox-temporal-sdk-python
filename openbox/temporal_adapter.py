"""Temporal enforcement adapter for the shared OpenBox runtime.

Imported only by Worker/plugin composition code, never by workflow-safe helper
modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

from openbox_core.contracts.results import EvaluationResult, Verdict

if TYPE_CHECKING:
    from openbox_core.contracts.context import ActivityContext


class TemporalFrameworkAdapter:
    """Translate shared governance results into Temporal-native failures."""

    name = "temporal"

    async def handle_approval(
        self,
        result: EvaluationResult,
        context: "ActivityContext | None" = None,
    ) -> None:
        from temporalio.exceptions import ApplicationError

        raise ApplicationError(
            result.reason or "Approval required",
            type="GovernanceApprovalPending",
            non_retryable=False,
        )

    def raise_lifecycle_blocked(self, result: EvaluationResult) -> NoReturn:
        self._raise_native(result)

    def raise_hook_blocked(self, result: EvaluationResult) -> NoReturn:
        self._raise_native(result)

    def on_completed_hook_result(
        self,
        result: EvaluationResult,
        context: "ActivityContext | None" = None,
    ) -> None:
        # Completed work cannot be undone. Temporal-specific correlation state
        # remains owned by WorkflowSpanProcessor/hook_governance.
        return None

    @staticmethod
    def _raise_native(result: EvaluationResult) -> NoReturn:
        from temporalio.exceptions import ApplicationError

        error_type = (
            "GovernanceHalt" if result.verdict is Verdict.HALT else "GovernanceBlock"
        )
        raise ApplicationError(
            result.reason or "Blocked by governance",
            type=error_type,
            non_retryable=True,
        )
