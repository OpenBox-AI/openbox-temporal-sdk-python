"""Deterministic Workflow helper for one-attempt governed commands.

This module performs no networking, signing, logging, sandbox I/O, or argv mapping.
Workflow history contains only the bounded structured request.
"""

from __future__ import annotations

from datetime import timedelta
from typing import cast

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.workflow import ActivityCancellationType

from .sandbox.types import (
    GOVERNED_COMMAND_ACTIVITY_TYPE,
    GovernedCommandActivityResult,
    GovernedCommandRequest,
)


async def execute_governed_command(
    request: GovernedCommandRequest,
    *,
    start_to_close_timeout: timedelta = timedelta(minutes=6),
    heartbeat_timeout: timedelta = timedelta(minutes=2),
) -> GovernedCommandActivityResult:
    if not isinstance(request, GovernedCommandRequest):
        raise TypeError("execute_governed_command requires GovernedCommandRequest")
    if start_to_close_timeout <= timedelta(0):
        raise ValueError("start_to_close_timeout must be positive")
    if heartbeat_timeout <= timedelta(0):
        raise ValueError("heartbeat_timeout must be positive")
    return cast(
        GovernedCommandActivityResult,
        await workflow.execute_activity(
            GOVERNED_COMMAND_ACTIVITY_TYPE,
            request.to_history_value(),
            result_type=GovernedCommandActivityResult,
            start_to_close_timeout=start_to_close_timeout,
            heartbeat_timeout=heartbeat_timeout,
            cancellation_type=ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
            retry_policy=RetryPolicy(maximum_attempts=1),
        ),
    )
