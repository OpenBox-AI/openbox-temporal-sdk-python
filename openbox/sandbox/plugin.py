"""Command-only Temporal plugin backed by openbox_sandbox."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import cast

from temporalio.plugin import SimplePlugin
from temporalio.worker import WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

from ..governed_command_activity import governed_command_activity
from .adapter import TemporalSandboxConfig
from .interceptor import GovernedCommandInterceptor


class OpenBoxSandboxPlugin(SimplePlugin):
    """Register the governed-command interceptor without constructing Core."""

    def __init__(self, sandbox: TemporalSandboxConfig):
        if not isinstance(sandbox, TemporalSandboxConfig):
            raise TypeError("TemporalSandboxConfig required")

        def workflow_runner(runner: WorkflowRunner | None) -> WorkflowRunner | None:
            if runner is None or not isinstance(runner, SandboxedWorkflowRunner):
                return runner
            return dataclasses.replace(
                runner,
                restrictions=runner.restrictions.with_passthrough_modules(
                    "openbox_sandbox"
                ),
            )

        super().__init__(
            "openbox.OpenBoxSandboxPlugin",
            interceptors=[GovernedCommandInterceptor(sandbox)],
            activities=[governed_command_activity],
            workflow_runner=cast(
                Callable[[WorkflowRunner | None], WorkflowRunner],
                workflow_runner,
            ),
        )
