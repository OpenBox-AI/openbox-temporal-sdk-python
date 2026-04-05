# openbox/plugin.py
"""
OpenBox Plugin for Temporal Workers.

Provides OpenBoxPlugin(SimplePlugin) — a drop-in plugin for Temporal's
AI Partner Ecosystem. Adds governance, observability, and hook-level
policy enforcement to Temporal workflows.

Usage:
    from openbox.plugin import OpenBoxPlugin

    worker = Worker(
        client, task_queue="q",
        workflows=[MyWorkflow], activities=[my_activity],
        plugins=[OpenBoxPlugin(openbox_url=..., openbox_api_key=...)],
    )
"""

import dataclasses
from typing import Any, Optional, Set

from temporalio.plugin import SimplePlugin
from temporalio.worker import WorkerConfig, WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

from .config import initialize as validate_api_key, GovernanceConfig
from .span_processor import WorkflowSpanProcessor
from .client import GovernanceClient


class OpenBoxPlugin(SimplePlugin):
    """Temporal Plugin for OpenBox governance and observability.

    Drop-in replacement for create_openbox_worker(). Registers governance
    interceptors, OTel instrumentation, and the send_governance_event activity.

    Example:
        worker = Worker(
            client, task_queue="my-queue",
            workflows=[MyWorkflow], activities=[my_activity],
            plugins=[OpenBoxPlugin(
                openbox_url="https://api.openbox.ai",
                openbox_api_key="obx_live_...",
            )],
        )
    """

    def __init__(
        self,
        *,
        openbox_url: str,
        openbox_api_key: str,
        governance_timeout: float = 30.0,
        governance_policy: str = "fail_open",
        send_start_event: bool = True,
        send_activity_start_event: bool = True,
        skip_workflow_types: Optional[Set[str]] = None,
        skip_activity_types: Optional[Set[str]] = None,
        skip_signals: Optional[Set[str]] = None,
        hitl_enabled: bool = True,
        instrument_databases: bool = True,
        db_libraries: Optional[Set[str]] = None,
        sqlalchemy_engine: Optional[Any] = None,
        instrument_file_io: bool = True,
    ):
        # 1. Validate API key (sync, uses urllib)
        validate_api_key(
            api_url=openbox_url,
            api_key=openbox_api_key,
            governance_timeout=governance_timeout,
        )

        # 2. Create span processor
        self._span_processor = WorkflowSpanProcessor(
            ignored_url_prefixes=[openbox_url]
        )

        # 3. Setup OTel instrumentation (HTTP, DB, File I/O)
        from .otel_setup import setup_opentelemetry_for_governance

        setup_opentelemetry_for_governance(
            self._span_processor,
            api_url=openbox_url,
            api_key=openbox_api_key,
            ignored_urls=[openbox_url],
            instrument_databases=instrument_databases,
            db_libraries=db_libraries,
            instrument_file_io=instrument_file_io,
            sqlalchemy_engine=sqlalchemy_engine,
            api_timeout=governance_timeout,
            on_api_error=governance_policy,
        )

        # 4. Create governance config
        config = GovernanceConfig(
            on_api_error=governance_policy,
            api_timeout=governance_timeout,
            send_start_event=send_start_event,
            send_activity_start_event=send_activity_start_event,
            skip_workflow_types=skip_workflow_types or set(),
            skip_activity_types=skip_activity_types or {"send_governance_event"},
            skip_signals=skip_signals or set(),
            hitl_enabled=hitl_enabled,
        )

        # 5. Create interceptors
        from .workflow_interceptor import GovernanceInterceptor
        from .activity_interceptor import ActivityGovernanceInterceptor

        governance_client = GovernanceClient(
            api_url=openbox_url,
            api_key=openbox_api_key,
            timeout=governance_timeout,
            on_api_error=governance_policy,
        )

        interceptors = [
            GovernanceInterceptor(
                api_url=openbox_url,
                api_key=openbox_api_key,
                span_processor=self._span_processor,
                config=config,
            ),
            ActivityGovernanceInterceptor(
                api_url=openbox_url,
                api_key=openbox_api_key,
                span_processor=self._span_processor,
                config=config,
                client=governance_client,
            ),
        ]

        # 6. Get governance activity
        from .activities import send_governance_event

        # 7. Sandbox passthrough for opentelemetry
        def workflow_runner(runner: WorkflowRunner | None) -> WorkflowRunner | None:
            if runner is None:
                return None
            if isinstance(runner, SandboxedWorkflowRunner):
                return dataclasses.replace(
                    runner,
                    restrictions=runner.restrictions.with_passthrough_modules(
                        "opentelemetry"
                    ),
                )
            return runner

        # Store config for logging
        self._governance_policy = governance_policy
        self._governance_timeout = governance_timeout
        self._instrument_databases = instrument_databases
        self._instrument_file_io = instrument_file_io
        self._hitl_enabled = hitl_enabled

        super().__init__(
            "openbox.OpenBoxPlugin",
            interceptors=interceptors,
            activities=[send_governance_event],
            workflow_runner=workflow_runner,
        )

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        """Store Temporal client ref for HALT terminate calls, then delegate."""
        from .activities import set_temporal_client

        client = config.get("client")
        if client:
            set_temporal_client(client)

        config = super().configure_worker(config)

        print("OpenBox Plugin initialized successfully")
        print(f"  - Governance policy: {self._governance_policy}")
        print(f"  - Governance timeout: {self._governance_timeout}s")
        db_status = "enabled" if self._instrument_databases else "disabled"
        file_status = "enabled" if self._instrument_file_io else "disabled"
        hitl_status = "enabled" if self._hitl_enabled else "disabled"
        print(f"  - Database instrumentation: {db_status}")
        print(f"  - File I/O instrumentation: {file_status}")
        print(f"  - Approval polling: {hitl_status}")
        print("  - Hook governance: enabled")

        return config
