"""Dedicated command-only Temporal Worker composition."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from temporalio.client import Client
from temporalio.worker import Interceptor, Worker

from ..governed_command_activity import governed_command_activity
from .adapter import TemporalSandboxConfig
from .interceptor import GovernedCommandInterceptor


def create_sandbox_worker(
    client: Client,
    task_queue: str,
    *,
    sandbox: TemporalSandboxConfig,
    interceptors: Sequence[Interceptor] = (),
    **worker_options: Any,
) -> Worker:
    """Create a Worker that owns sandbox execution but no Core client.

    The application agent authorizes before Workflow start. This Worker accepts
    only the governed-command Activity and either an explicit same-domain trust
    configuration or a verified receipt.
    """
    forbidden = {
        "activities",
        "plugins",
        "workflow_runner",
        "workflows",
    } & set(worker_options)
    if forbidden:
        raise TypeError(f"reserved Worker options: {sorted(forbidden)}")
    return Worker(
        client,
        task_queue=task_queue,
        workflows=[],
        activities=[governed_command_activity],
        interceptors=[GovernedCommandInterceptor(sandbox), *interceptors],
        **worker_options,
    )
