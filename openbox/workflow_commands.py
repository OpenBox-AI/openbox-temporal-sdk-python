"""Deterministic Workflow helper for one-attempt sandbox commands."""

from __future__ import annotations

from datetime import timedelta
from typing import cast

from openbox_sandbox.contracts import GOVERNED_COMMAND_ACTIVITY_TYPE
from openbox_sandbox.contracts import (
    GovernedCommandActivityResult as SandboxActivityResult,
)
from openbox_sandbox.contracts import GovernedCommandRequest as SandboxCommandRequest
from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.workflow import ActivityCancellationType


async def execute_governed_command(
    request: SandboxCommandRequest,
    *,
    start_to_close_timeout: timedelta = timedelta(minutes=6),
    heartbeat_timeout: timedelta = timedelta(minutes=2),
) -> SandboxActivityResult:
    """Schedule exactly one governed-command Activity attempt."""
    if not isinstance(request, SandboxCommandRequest):
        raise TypeError("execute_governed_command requires SandboxCommandRequest")
    if start_to_close_timeout <= timedelta(0):
        raise ValueError("start_to_close_timeout must be positive")
    if heartbeat_timeout <= timedelta(0):
        raise ValueError("heartbeat_timeout must be positive")
    return cast(
        SandboxActivityResult,
        await workflow.execute_activity(
            GOVERNED_COMMAND_ACTIVITY_TYPE,
            request.to_history_value(),
            result_type=SandboxActivityResult,
            start_to_close_timeout=start_to_close_timeout,
            heartbeat_timeout=heartbeat_timeout,
            cancellation_type=ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
            retry_policy=RetryPolicy(maximum_attempts=1),
        ),
    )
