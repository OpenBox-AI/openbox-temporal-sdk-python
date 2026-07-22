"""Defensive body for the governed-command Activity registration."""

from openbox_sandbox import GOVERNED_COMMAND_ACTIVITY_TYPE, SandboxCommandRequest
from temporalio import activity
from temporalio.exceptions import ApplicationError


@activity.defn(name=GOVERNED_COMMAND_ACTIVITY_TYPE)
async def governed_command_activity(request: SandboxCommandRequest) -> None:
    del request
    raise ApplicationError(
        "Governed command reached its defensive Activity body",
        type="GovernedCommandInterceptorRequired",
        non_retryable=True,
    )
