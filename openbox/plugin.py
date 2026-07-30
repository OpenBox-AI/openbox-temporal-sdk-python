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
import logging
from typing import Any, Optional, Set

from temporalio.plugin import SimplePlugin
from temporalio.worker import WorkerConfig, WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

from .client import GovernanceClient
from .config import GovernanceConfig
from .config import initialize as validate_api_key

logger = logging.getLogger(__name__)


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
        agent_did: Optional[str] = None,
        agent_private_key: Optional[str] = None,
        openbox_agent_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        deployment_id: Optional[str] = None,
        okta_agent_id: Optional[str] = None,
        okta_agent_key_id: Optional[str] = None,
        okta_agent_private_key: Optional[str] = None,
        okta_agent_algorithm: Optional[str] = None,
        agent_proof_audience: Optional[str] = None,
        governance_timeout: float = 30.0,
        governance_policy: str = "fail_open",
        send_start_event: bool = True,
        send_activity_start_event: bool = True,
        skip_workflow_types: Optional[Set[str]] = None,
        skip_activity_types: Optional[Set[str]] = None,
        skip_signals: Optional[Set[str]] = None,
        hitl_enabled: bool = True,
        max_patch_restarts: int = 3,
        instrument_databases: bool = True,
        db_libraries: Optional[Set[str]] = None,
        sqlalchemy_engine: Optional[Any] = None,
        instrument_file_io: bool = True,
        enable_trace_propagation: bool = True,
    ):
        """OpenBox governance plugin for a Temporal ``Worker``.

        agent_did / agent_private_key: OpenBox DID identity (v1). Both-or-neither.
        openbox_agent_id / organization_id / deployment_id / okta_agent_id /
        okta_agent_key_id / okta_agent_private_key / okta_agent_algorithm /
        agent_proof_audience: Okta AI Agent identity (v2, proposal §13.7).
        All-or-nothing together, and mutually exclusive with agent_did/
        agent_private_key — at most one identity verification method.
        """
        validate_api_key(
            api_url=openbox_url,
            api_key=openbox_api_key,
            governance_timeout=governance_timeout,
            agent_did=agent_did,
            agent_private_key=agent_private_key,
            openbox_agent_id=openbox_agent_id,
            organization_id=organization_id,
            deployment_id=deployment_id,
            okta_agent_id=okta_agent_id,
            okta_agent_key_id=okta_agent_key_id,
            okta_agent_private_key=okta_agent_private_key,
            okta_agent_algorithm=okta_agent_algorithm,
            agent_proof_audience=agent_proof_audience,
        )

        from .config import get_global_config

        _signer = get_global_config().get_signer()
        _okta_identity = get_global_config().get_okta_identity()

        from .governance_state import TemporalGovernanceState

        self._state = TemporalGovernanceState()

        from .core_adapter import create_core_runtime

        self._runtime = create_core_runtime(
            api_url=openbox_url,
            api_key=openbox_api_key,
            state=self._state,
            timeout_seconds=governance_timeout,
            on_api_error=governance_policy,
            agent_did=agent_did,
            agent_private_key=agent_private_key,
            openbox_agent_id=openbox_agent_id,
            organization_id=organization_id,
            deployment_id=deployment_id,
            okta_agent_id=okta_agent_id,
            okta_agent_key_id=okta_agent_key_id,
            okta_agent_private_key=okta_agent_private_key,
            okta_agent_algorithm=okta_agent_algorithm,
            agent_proof_audience=agent_proof_audience,
            hitl_enabled=hitl_enabled,
            skip_workflow_types=skip_workflow_types or set(),
            skip_activity_types=skip_activity_types or {"send_governance_event"},
            skip_signals=skip_signals or set(),
            send_start_event=send_start_event,
            send_activity_start_event=send_activity_start_event,
            instrument_databases=instrument_databases,
            instrument_file_io=instrument_file_io,
        )
        self._runtime.install_instrumentation()

        config = GovernanceConfig(
            on_api_error=governance_policy,
            api_timeout=governance_timeout,
            send_start_event=send_start_event,
            send_activity_start_event=send_activity_start_event,
            skip_workflow_types=skip_workflow_types or set(),
            skip_activity_types=skip_activity_types or {"send_governance_event"},
            skip_signals=skip_signals or set(),
            hitl_enabled=hitl_enabled,
            max_patch_restarts=max_patch_restarts,
        )

        from .activity_interceptor import ActivityGovernanceInterceptor
        from .workflow_interceptor import GovernanceInterceptor

        governance_client = GovernanceClient(
            api_url=openbox_url,
            api_key=openbox_api_key,
            timeout=governance_timeout,
            on_api_error=governance_policy,
            agent_did=agent_did,
            signer=_signer,
            okta_identity=_okta_identity,
        )

        interceptors: list = [
            GovernanceInterceptor(
                api_url=openbox_url,
                api_key=openbox_api_key,
                state=self._state,
                config=config,
            ),
            ActivityGovernanceInterceptor(
                api_url=openbox_url,
                api_key=openbox_api_key,
                state=self._state,
                config=config,
                client=governance_client,
            ),
        ]

        if enable_trace_propagation:
            from temporalio.contrib.opentelemetry import TracingInterceptor

            interceptors.append(TracingInterceptor())

        from .activities import build_governance_activities

        governance_activities = build_governance_activities(
            api_url=openbox_url,
            api_key=openbox_api_key,
            agent_did=agent_did,
            signer=_signer,
            okta_identity=_okta_identity,
        )

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

        self._governance_policy = governance_policy
        self._governance_timeout = governance_timeout
        self._instrument_databases = instrument_databases
        self._instrument_file_io = instrument_file_io
        self._hitl_enabled = hitl_enabled

        super().__init__(
            "openbox.OpenBoxPlugin",
            interceptors=interceptors,
            activities=[governance_activities.send_governance_event],
            workflow_runner=workflow_runner,
        )

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        """Store Temporal client ref for HALT terminate calls, then delegate."""
        from .activities import set_temporal_client

        client = config.get("client")
        if client:
            set_temporal_client(client)

        config = super().configure_worker(config)

        db_status = "enabled" if self._instrument_databases else "disabled"
        file_status = "enabled" if self._instrument_file_io else "disabled"
        hitl_status = "enabled" if self._hitl_enabled else "disabled"
        logger.info(
            "OpenBox Plugin initialized: policy=%s timeout=%ss "
            "db=%s file=%s hitl=%s instrumentation=openbox_core",
            self._governance_policy,
            self._governance_timeout,
            db_status,
            file_status,
            hitl_status,
        )

        return config
