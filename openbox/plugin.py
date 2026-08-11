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

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from temporalio.plugin import SimplePlugin
from temporalio.worker import Worker, WorkerConfig, WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

from .client import GovernanceClient
from .config import GovernanceConfig
from .config import initialize as validate_api_key
from .span_processor import WorkflowSpanProcessor

if TYPE_CHECKING:  # pragma: no cover — type-only reference
    from .sandbox.adapter import (
        TemporalSandboxConfig,
    )

logger = logging.getLogger(__name__)


class OpenBoxPlugin(SimplePlugin):
    """Temporal Plugin for OpenBox governance and observability.

    Registers governance
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
        # AIP DID + Ed25519 signing (both-or-neither). Required for
        # signing_required=true agents; every Core request is signed locally.
        agent_did: str | None = None,
        agent_private_key: str | None = None,
        core_ca_path: str | None = None,
        governance_timeout: float = 30.0,
        governance_policy: str = "fail_open",
        send_start_event: bool = True,
        send_activity_start_event: bool = True,
        skip_workflow_types: set[str] | None = None,
        skip_activity_types: set[str] | None = None,
        skip_signals: set[str] | None = None,
        hitl_enabled: bool = True,
        max_patch_restarts: int = 3,
        sandbox: TemporalSandboxConfig | None = None,
        instrument_http: bool = True,

        instrument_databases: bool = True,
        db_libraries: set[str] | None = None,
        sqlalchemy_engine: Any | None = None,
        instrument_file_io: bool = True,
        # Propagate W3C traceparent/baggage through Temporal headers so spans
        # started by the caller (e.g., an HTTP server) stitch to workflow and
        # activity spans on the worker side. Uses Temporal's built-in
        # TracingInterceptor under the hood.
        enable_trace_propagation: bool = True,
        add_temporal_tracing: bool = True,
    ):
        from .sandbox.adapter import require_matching_governance_signing

        require_matching_governance_signing(sandbox, agent_did)

        # 1. Validate API key (sync, uses urllib). Also loads the Ed25519 signer
        #    and validates a signing_required=true agent via a signed GET.
        validate_api_key(
            api_url=openbox_url,
            api_key=openbox_api_key,
            governance_timeout=governance_timeout,
            agent_did=agent_did,
            agent_private_key=agent_private_key,
            core_ca_path=core_ca_path,
        )
        from .config import get_global_config
        from .governance_state import TemporalGovernanceState

        global_config: Any = get_global_config()
        _signer: Any = global_config.get_signer()

        # One run-scoped state shared by the runtime adapter and both
        # interceptors: signal verdicts (workflow records → activity enforces),
        # HITL pending-approval markers, and completed-hook stops all flow
        # through this single instance.
        self._state = TemporalGovernanceState()

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
        from .runtime import create_temporal_runtime

        self._runtime = create_temporal_runtime(
            config,
            api_url=openbox_url,
            api_key=openbox_api_key,
            timeout=governance_timeout,
            on_api_error=governance_policy,
            agent_did=agent_did,
            signer=_signer,
            core_ca_path=core_ca_path,
            state=self._state,
        )

        # 2. Create span processor
        self._span_processor = WorkflowSpanProcessor(ignored_url_prefixes=[openbox_url])

        # 3. Setup OTel instrumentation (HTTP, DB, File I/O)
        from .otel_setup import setup_opentelemetry_for_governance

        governed_otel_bridge = sandbox is not None and sandbox.otel_bridge is not None
        effective_instrument_http = instrument_http and not governed_otel_bridge
        effective_instrument_databases = (
            instrument_databases and not governed_otel_bridge
        )
        effective_instrument_file_io = instrument_file_io and not governed_otel_bridge
        setup_kwargs: dict[str, Any] = {
            "api_url": openbox_url,
            "api_key": openbox_api_key,
            "ignored_urls": [openbox_url],
            "instrument_http": effective_instrument_http,
            "instrument_databases": effective_instrument_databases,
            "db_libraries": db_libraries,
            "instrument_file_io": effective_instrument_file_io,
            "sqlalchemy_engine": sqlalchemy_engine,
            "api_timeout": governance_timeout,
            "on_api_error": governance_policy,
            "max_body_size": 65536,
            "agent_did": agent_did,
            "signer": _signer,
            "core_ca_path": core_ca_path,
        }
        if governed_otel_bridge:
            setup_kwargs["register_span_processor"] = False
        setup_opentelemetry_for_governance(self._span_processor, **setup_kwargs)
        from . import hook_governance as _hook_governance

        _hook_governance.set_evaluation_client(self._runtime.client)

        # 5. Create interceptors
        from .activity_interceptor import ActivityGovernanceInterceptor
        from .workflow_interceptor import GovernanceInterceptor

        governance_client = GovernanceClient._from_core_client(
            self._runtime.client,
            api_url=openbox_url,
            api_key=openbox_api_key,
            timeout=governance_timeout,
            on_api_error=governance_policy,
            agent_did=agent_did,
            signer=_signer,
            ssl_context=global_config.get_ssl_context(),
        )

        self._activity_interceptor = ActivityGovernanceInterceptor(
            api_url=openbox_url,
            api_key=openbox_api_key,
            span_processor=self._span_processor,
            config=config,
            client=governance_client,
            sandbox=sandbox,
            state=self._state,
        )
        interceptors: list[Any] = [
            self._activity_interceptor,
            GovernanceInterceptor(
                api_url=openbox_url,
                api_key=openbox_api_key,
                state=self._state,
                config=config,
            ),
        ]

        # Header-based OTel trace propagation. Temporal's built-in
        # TracingInterceptor implements both client.Interceptor and
        # worker.Interceptor — SimplePlugin auto-routes it to both sides.
        # Without it, trace IDs set by the caller don't reach workflow/
        # activity spans, leaving disconnected trees in the backend. The
        # interceptor is passive (no instrumentation, no span processors).
        # Callers that compose the Worker themselves (e.g. with a governed-
        # command OTEL bridge) may pass add_temporal_tracing=False and supply
        # their own TracingInterceptor via Worker(interceptors=...).
        if enable_trace_propagation and add_temporal_tracing:
            from temporalio.contrib.opentelemetry import TracingInterceptor

            interceptors.insert(0, TracingInterceptor())

        # 6. Build governance activity instance with credentials captured in self —
        # avoids passing api_key through activity inputs / workflow history.
        from .activities import build_governance_activities

        governance_activities = build_governance_activities(
            api_url=openbox_url,
            api_key=openbox_api_key,
            agent_did=agent_did,
            signer=_signer,
            core_ca_path=core_ca_path,
        )
        governance_activities._governance_client = governance_client

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
        self._instrument_http = effective_instrument_http
        self._instrument_databases = effective_instrument_databases
        self._instrument_file_io = effective_instrument_file_io
        self._hitl_enabled = hitl_enabled

        plugin_activities: list[Callable[..., Any]] = [
            governance_activities.send_governance_event
        ]
        self._sandbox = sandbox
        self._otel_bridge = None if sandbox is None else sandbox.otel_bridge
        if sandbox is not None:
            from .governed_command_activity import governed_command_activity

            plugin_activities.append(governed_command_activity)

        super().__init__(
            "openbox.OpenBoxPlugin",
            interceptors=interceptors,
            activities=plugin_activities,
            workflow_runner=workflow_runner,  # type: ignore[arg-type]
        )

    async def run_worker(
        self, worker: Worker, next: Callable[[Worker], Awaitable[None]]
    ) -> None:
        """Bound the optional v3 daemon to Worker execution, never replay."""
        if self._otel_bridge is not None:
            self._otel_bridge.start()
        try:
            await super().run_worker(worker, next)
        finally:
            if self._otel_bridge is not None:
                # The bridge owns a fixed bounded join; do not queue shutdown
                # behind a potentially saturated default executor.
                self._otel_bridge.shutdown()
            runtime = getattr(self, "_runtime", None)
            if runtime is not None:
                await runtime.aclose()

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        """Store Temporal client ref for HALT terminate calls, then delegate."""
        from .activities import set_temporal_client

        client = config.get("client")
        if client:
            set_temporal_client(client)

        config = super().configure_worker(config)
        if self._sandbox is not None:
            governed_heartbeat = timedelta(
                seconds=self._sandbox.heartbeat_interval_seconds
            )
            config["max_heartbeat_throttle_interval"] = governed_heartbeat
            config["default_heartbeat_throttle_interval"] = governed_heartbeat
            configured = list(config.get("interceptors") or [])
            if self._activity_interceptor not in configured:
                raise ValueError("OpenBox Activity interceptor was not installed")
            config["interceptors"] = [
                self._activity_interceptor,
                *(
                    item
                    for item in configured
                    if item is not self._activity_interceptor
                ),
            ]

        http_status = (
            "enabled" if self._instrument_http else "disabled (no uninstallation)"
        )
        db_status = "enabled" if self._instrument_databases else "disabled"
        file_status = "enabled" if self._instrument_file_io else "disabled"
        hitl_status = "enabled" if self._hitl_enabled else "disabled"
        logger.info(
            "OpenBox Plugin initialized: policy=%s timeout=%ss "
            "http=%s db=%s file=%s hitl=%s",
            self._governance_policy,
            self._governance_timeout,
            http_status,
            db_status,
            file_status,
            hitl_status,
        )

        return config
