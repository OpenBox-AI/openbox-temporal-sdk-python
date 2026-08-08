"""Defensive Activity body for registered governed commands."""

from temporalio import activity
from temporalio.exceptions import ApplicationError

from .sandbox.types import GOVERNED_COMMAND_ACTIVITY_TYPE, GovernedCommandRequest


@activity.defn(name=GOVERNED_COMMAND_ACTIVITY_TYPE)
async def governed_command_activity(request: GovernedCommandRequest) -> None:
    del request
    raise ApplicationError(
        "Governed command reached its defensive Activity body",
        type="GovernedCommandInterceptorRequired",
        non_retryable=True,
    )
