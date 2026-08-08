from __future__ import annotations
# openbox/worker.py
"""
OpenBox-enabled Temporal Worker factory.

Provides a simple function to create a Temporal Worker with all OpenBox
governance components pre-configured.

Usage:
    from openbox import create_openbox_worker

    worker = await create_openbox_worker(
        client=client,
        task_queue="my-queue",
        workflows=[MyWorkflow],
        activities=[my_activity],
    )

    await worker.run()
"""

import logging
from concurrent.futures import Executor, ThreadPoolExecutor
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional, Sequence

from temporalio.client import Client
from temporalio.plugin import SimplePlugin
from temporalio.worker import Interceptor, Worker, WorkerDeploymentConfig

from .config import GovernanceConfig
from .config import initialize as validate_api_key
if TYPE_CHECKING:  # pragma: no cover — type-only reference
    from .sandbox.adapter import (
        TemporalSandboxConfig,
        require_matching_governance_signing,
    )
from .span_processor import WorkflowSpanProcessor

logger = logging.getLogger(__name__)


class _OpenBoxRuntimePlugin(SimplePlugin):
    """Own and close the one shared runtime for a normal connected Worker."""

    def __init__(self, runtime: Any) -> None:
        super().__init__("openbox.OpenBoxRuntimePlugin")
        self.runtime = runtime

    async def run_worker(
        self, worker: Worker, next: Callable[[Worker], Awaitable[None]]
    ) -> None:
        try:
            await next(worker)
        finally:
            await self.runtime.aclose()


class _GovernedCommandTelemetryPlugin(SimplePlugin):
    """Bind an explicit sandbox's v3 bridge to Worker execution."""

    def __init__(self, bridge: Any) -> None:
        super().__init__("openbox.GovernedCommandTelemetryPlugin")
        self._bridge = bridge

    async def run_worker(
        self, worker: Worker, next: Callable[[Worker], Awaitable[None]]
    ) -> None:
        self._bridge.start()
        try:
            await next(worker)
        finally:
            self._bridge.shutdown()


def create_openbox_worker(
    client: Client,
    task_queue: str,
    *,
    workflows: Sequence[type[Any]] = (),
    activities: Sequence[Callable[..., Any]] = (),
    # OpenBox config (required)
    openbox_url: str,
    openbox_api_key: str,
    # AIP DID + Ed25519 signing (both-or-neither). When set, every Core request
    # is signed locally; required for signing_required=true agents.
    agent_did: Optional[str] = None,
    agent_private_key: Optional[str] = None,
    core_ca_path: Optional[str] = None,
    governance_timeout: float = 30.0,
    governance_policy: str = "fail_open",
    send_start_event: bool = True,
    send_activity_start_event: bool = True,
    skip_workflow_types: Optional[set[str]] = None,
    skip_activity_types: Optional[set[str]] = None,
    skip_signals: Optional[set[str]] = None,
    # HITL configuration
    hitl_enabled: bool = True,
    max_patch_restarts: int = 3,
    # Registered governed-command configuration (strict and fail closed)
    sandbox: Optional[TemporalSandboxConfig] = None,
    # Global HTTP instrumentation and hook-level HTTP capture
    instrument_http: bool = True,
    # Database instrumentation
    instrument_databases: bool = True,
    db_libraries: Optional[set[str]] = None,
    sqlalchemy_engine: Optional[Any] = None,
    # File I/O instrumentation
    instrument_file_io: bool = True,
    # Header-based W3C trace propagation via Temporal's built-in TracingInterceptor.
    # Without it, trace IDs set by the caller don't reach workflow/activity spans.
    enable_trace_propagation: bool = True,
    # Standard Worker options
    add_temporal_tracing: bool = True,
    activity_executor: Optional[Executor] = None,
    workflow_task_executor: Optional[ThreadPoolExecutor] = None,
    interceptors: Sequence[Interceptor] = (),
    plugins: Sequence[Any] = (),
    build_id: Optional[str] = None,
    identity: Optional[str] = None,
    max_cached_workflows: int = 1000,
    max_concurrent_workflow_tasks: Optional[int] = None,
    max_concurrent_activities: Optional[int] = None,
    max_concurrent_local_activities: Optional[int] = None,
    max_concurrent_workflow_task_polls: int = 5,
    nonsticky_to_sticky_poll_ratio: float = 0.2,
    max_concurrent_activity_task_polls: int = 5,
    no_remote_activities: bool = False,
    sticky_queue_schedule_to_start_timeout: timedelta = timedelta(seconds=10),
    max_heartbeat_throttle_interval: timedelta = timedelta(seconds=60),
    default_heartbeat_throttle_interval: timedelta = timedelta(seconds=30),
    max_activities_per_second: Optional[float] = None,
    max_task_queue_activities_per_second: Optional[float] = None,
    graceful_shutdown_timeout: timedelta = timedelta(),
    shared_state_manager: Any = None,
    debug_mode: bool = False,
    disable_eager_activity_execution: bool = False,
    on_fatal_error: Optional[Callable[[BaseException], Awaitable[None]]] = None,
    use_worker_versioning: bool = False,
    disable_safe_workflow_eviction: bool = False,
    deployment_config: Optional[WorkerDeploymentConfig] = None,
) -> Worker:
    """
    Create a Temporal Worker with OpenBox governance enabled.

    This function:
    1. Validates the OpenBox API key
    2. Sets up OpenTelemetry HTTP instrumentation
    3. Creates governance interceptors
    4. Returns a fully configured Worker

    Args:
        client: Temporal client
        task_queue: Task queue name
        workflows: List of workflow classes
        activities: List of activity functions (OpenBox activities added automatically)

        # OpenBox config
        openbox_url: OpenBox Core API URL (required for governance)
        openbox_api_key: OpenBox API key (required for governance)
        core_ca_path: Optional CA bundle used to pin all Core HTTPS clients
        governance_timeout: Timeout for governance API calls (default: 30.0s)
        governance_policy: "fail_open" or "fail_closed" (default: "fail_open")
        send_start_event: Send WorkflowStarted events (default: True)
        send_activity_start_event: Send ActivityStarted events (default: True)
        skip_workflow_types: Workflow types to skip governance
        skip_activity_types: Activity types to skip governance
        skip_signals: Signal names to skip governance

        # HTTP instrumentation
        instrument_http: Install process-global HTTP instrumentation and capture
            hooks (default: True). False does not uninstall hooks installed earlier.

        # Database instrumentation
        instrument_databases: Instrument database libraries (default: True)
        db_libraries: Set of database libraries to instrument (None = all available).
                      Valid values: "psycopg2", "asyncpg", "mysql", "pymysql",
                      "pymongo", "redis", "sqlalchemy"
        sqlalchemy_engine: SQLAlchemy Engine instance to instrument. Pass this when
                          the engine is created before create_openbox_worker() runs
                          (e.g., at module import time). This ensures query-level
                          instrumentation works on pre-existing engines.

        # File I/O instrumentation
        instrument_file_io: Instrument file I/O operations (default: False)

        # Standard Worker options (passed through to Worker)
        activity_executor: Executor for activities
        interceptors: Additional interceptors (OpenBox interceptors added automatically)
        plugins: Additional Temporal Worker plugins, preserved in caller order.
        deployment_config: Modern Temporal Worker Deployment configuration. Cannot
            be combined with legacy build_id or use_worker_versioning.
        ... (all other standard Worker options)

    Returns:
        Configured Worker instance

    Example:
        ```python
        import os
        from openbox import create_openbox_worker

        client = await Client.connect("localhost:7233")

        # Use HTTPS for production, HTTP is only allowed for localhost
        worker = create_openbox_worker(
            client=client,
            task_queue="my-queue",
            workflows=[MyWorkflow],
            activities=[my_activity, another_activity],
            openbox_url=os.getenv("OPENBOX_URL"),  # e.g., "https://api.openbox.ai"
            openbox_api_key=os.getenv("OPENBOX_API_KEY"),
            governance_policy="fail_closed",
        )

        await worker.run()
        ```
    """
    if deployment_config is not None and (
        build_id is not None or use_worker_versioning
    ):
        raise ValueError(
            "deployment_config cannot be combined with build_id or "
            "use_worker_versioning"
        )

    logger.info("Initializing OpenBox SDK with URL: %s", openbox_url)

    from .sandbox.adapter import require_matching_governance_signing

    require_matching_governance_signing(sandbox, agent_did)
    receipt_authorized = sandbox is not None and sandbox.receipt_verifier is not None
    trusted_application = sandbox is not None and sandbox.trust_application_agent
    disconnected_command_worker = receipt_authorized or trusted_application
    if disconnected_command_worker and (
        agent_did is not None or agent_private_key is not None
    ):
        raise ValueError("disconnected command workers do not hold Core signing keys")

    # 0. Store Temporal client reference for HALT terminate calls
    from .activities import set_temporal_client

    set_temporal_client(client)

    if disconnected_command_worker:
        # Branch before any shared config/client/runtime or instrumentation is
        # constructed. Only the command interceptor/activity and optional
        # existing-provider telemetry bridge are installed.
        from .activity_interceptor import ActivityGovernanceInterceptor
        from .governed_command_activity import governed_command_activity

        command_interceptor = ActivityGovernanceInterceptor(
            api_url=openbox_url,
            api_key=openbox_api_key,
            span_processor=None,  # type: ignore[arg-type]
            config=GovernanceConfig(),
            client=None,
            sandbox=sandbox,
        )
        command_plugins = list(plugins)
        if sandbox is not None and sandbox.otel_bridge is not None:
            command_plugins.append(_GovernedCommandTelemetryPlugin(sandbox.otel_bridge))
        command_interceptors: list[Interceptor] = [command_interceptor, *interceptors]
        if enable_trace_propagation:
            from temporalio.contrib.opentelemetry import TracingInterceptor

            # The governed-command interceptor executes its registered Activity
            # directly, so tracing must be outermost to establish current context.
            command_interceptors.insert(0, TracingInterceptor())
        assert sandbox is not None
        governed_heartbeat = timedelta(seconds=sandbox.heartbeat_interval_seconds)
        return Worker(
            client,
            task_queue=task_queue,
            workflows=workflows,
            activities=[*activities, governed_command_activity],
            activity_executor=activity_executor,
            workflow_task_executor=workflow_task_executor,
            plugins=command_plugins,
            interceptors=command_interceptors,
            build_id=build_id,
            identity=identity,
            max_cached_workflows=max_cached_workflows,
            max_concurrent_workflow_tasks=max_concurrent_workflow_tasks,
            max_concurrent_activities=max_concurrent_activities,
            max_concurrent_local_activities=max_concurrent_local_activities,
            max_concurrent_workflow_task_polls=max_concurrent_workflow_task_polls,
            nonsticky_to_sticky_poll_ratio=nonsticky_to_sticky_poll_ratio,
            max_concurrent_activity_task_polls=max_concurrent_activity_task_polls,
            no_remote_activities=no_remote_activities,
            sticky_queue_schedule_to_start_timeout=sticky_queue_schedule_to_start_timeout,
            max_heartbeat_throttle_interval=min(
                max_heartbeat_throttle_interval, governed_heartbeat
            ),
            default_heartbeat_throttle_interval=min(
                default_heartbeat_throttle_interval, governed_heartbeat
            ),
            max_activities_per_second=max_activities_per_second,
            max_task_queue_activities_per_second=max_task_queue_activities_per_second,
            graceful_shutdown_timeout=graceful_shutdown_timeout,
            shared_state_manager=shared_state_manager,
            debug_mode=debug_mode,
            disable_eager_activity_execution=disable_eager_activity_execution,
            on_fatal_error=on_fatal_error,
            use_worker_versioning=use_worker_versioning,
            disable_safe_workflow_eviction=disable_safe_workflow_eviction,
            deployment_config=deployment_config,
        )

    # 1. Connected mode validates Core credentials before constructing the one
    # shared runtime/client used by lifecycle and hook governance.
    validate_api_key(
        api_url=openbox_url,
        api_key=openbox_api_key,
        governance_timeout=governance_timeout,
        agent_did=agent_did,
        agent_private_key=agent_private_key,
        core_ca_path=core_ca_path,
    )
    from .config import get_global_config

    global_config: Any = get_global_config()
    _signer: Any = global_config.get_signer()

    # Map the unchanged Temporal dataclass into shared config groups, then own
    # one shared runtime/client for every connected decision path.
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
    from .runtime import create_temporal_runtime

    runtime = create_temporal_runtime(
        config,
        api_url=openbox_url,
        api_key=openbox_api_key,
        timeout=governance_timeout,
        on_api_error=governance_policy,
        agent_did=agent_did,
        signer=_signer,
        core_ca_path=core_ca_path,
    )

    # 2. Create span processor
    span_processor = WorkflowSpanProcessor(ignored_url_prefixes=[openbox_url])

    # 3. Setup OTel HTTP, database, and file I/O instrumentation
    from .otel_setup import setup_opentelemetry_for_governance

    governed_otel_bridge = sandbox is not None and sandbox.otel_bridge is not None
    effective_instrument_http = instrument_http and not governed_otel_bridge
    effective_instrument_databases = instrument_databases and not governed_otel_bridge
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
    setup_opentelemetry_for_governance(span_processor, **setup_kwargs)
    from . import hook_governance as _hook_governance

    _hook_governance.set_evaluation_client(runtime.client)

    # 5. Create interceptors
    from .activity_interceptor import ActivityGovernanceInterceptor
    from .client import GovernanceClient
    from .governance_state import TemporalGovernanceState
    from .workflow_interceptor import GovernanceInterceptor

    workflow_interceptor = GovernanceInterceptor(
        api_url=openbox_url,
        api_key=openbox_api_key,
        state=TemporalGovernanceState(),
        config=config,
    )

    # Temporal façade adapts the runtime's client; it does not construct another.
    governance_client = GovernanceClient._from_core_client(
        runtime.client,
        api_url=openbox_url,
        api_key=openbox_api_key,
        timeout=governance_timeout,
        on_api_error=governance_policy,
        agent_did=agent_did,
        signer=_signer,
        ssl_context=global_config.get_ssl_context(),
    )

    activity_interceptor = ActivityGovernanceInterceptor(
        api_url=openbox_url,
        api_key=openbox_api_key,
        span_processor=span_processor,
        config=config,
        client=governance_client,
        sandbox=sandbox,
    )

    # 6. Build governance activities with credentials captured on the instance
    # (so api_key never leaks through activity inputs into workflow history).
    from .activities import build_governance_activities

    governance_activities = build_governance_activities(
        api_url=openbox_url,
        api_key=openbox_api_key,
        agent_did=agent_did,
        signer=_signer,
        core_ca_path=core_ca_path,
    )
    governance_activities._governance_client = governance_client

    # The Activity interceptor is first because Temporal builds the inbound
    # chain in reverse; first in this list is observably outermost.
    all_interceptors: list[Interceptor] = [
        activity_interceptor,
        workflow_interceptor,
        *interceptors,
    ]

    # Header-based OTel trace propagation via Temporal's built-in interceptor.
    # The interceptor is passive and also runs with a governed-command OTEL
    # bridge, matching the OpenBoxPlugin composition.
    if enable_trace_propagation and add_temporal_tracing:
        from temporalio.contrib.opentelemetry import TracingInterceptor

        all_interceptors.append(TracingInterceptor())

    all_activities: list[Callable[..., Any]] = [
        *activities,
        governance_activities.send_governance_event,
    ]
    if sandbox is not None:
        from .governed_command_activity import governed_command_activity

        all_activities.append(governed_command_activity)

    logger.info(
        "OpenBox SDK initialized: policy=%s timeout=%ss http=%s db=%s file=%s "
        "hitl=%s events=WorkflowStarted,WorkflowCompleted,WorkflowFailed,"
        "SignalReceived,ActivityStarted,ActivityCompleted",
        governance_policy,
        governance_timeout,
        ("enabled" if effective_instrument_http else "disabled (no uninstallation)"),
        "enabled" if effective_instrument_databases else "disabled",
        "enabled" if effective_instrument_file_io else "disabled",
        "enabled" if hitl_enabled else "disabled",
    )

    # Registered-command cancellation is delivered through Temporal heartbeats.
    # The default 30–60 second throttles can otherwise let a short sandbox command
    # finish before cancellation reaches the Activity, so sandbox-enabled workers
    # clamp both throttles to the configured metadata-only heartbeat interval.
    if sandbox is not None:
        governed_heartbeat = timedelta(seconds=sandbox.heartbeat_interval_seconds)
        max_heartbeat_throttle_interval = min(
            max_heartbeat_throttle_interval, governed_heartbeat
        )
        default_heartbeat_throttle_interval = min(
            default_heartbeat_throttle_interval,
            max_heartbeat_throttle_interval,
        )

    all_plugins = list(plugins)
    if sandbox is not None and sandbox.otel_bridge is not None:
        all_plugins.append(_GovernedCommandTelemetryPlugin(sandbox.otel_bridge))
    all_plugins.append(_OpenBoxRuntimePlugin(runtime))

    # Create and return the public Temporal Worker type.
    return Worker(
        client,
        task_queue=task_queue,
        workflows=workflows,
        activities=all_activities,
        activity_executor=activity_executor,
        workflow_task_executor=workflow_task_executor,
        plugins=all_plugins,
        interceptors=all_interceptors,
        build_id=build_id,
        identity=identity,
        max_cached_workflows=max_cached_workflows,
        max_concurrent_workflow_tasks=max_concurrent_workflow_tasks,
        max_concurrent_activities=max_concurrent_activities,
        max_concurrent_local_activities=max_concurrent_local_activities,
        max_concurrent_workflow_task_polls=max_concurrent_workflow_task_polls,
        nonsticky_to_sticky_poll_ratio=nonsticky_to_sticky_poll_ratio,
        max_concurrent_activity_task_polls=max_concurrent_activity_task_polls,
        no_remote_activities=no_remote_activities,
        sticky_queue_schedule_to_start_timeout=sticky_queue_schedule_to_start_timeout,
        max_heartbeat_throttle_interval=max_heartbeat_throttle_interval,
        default_heartbeat_throttle_interval=default_heartbeat_throttle_interval,
        max_activities_per_second=max_activities_per_second,
        max_task_queue_activities_per_second=max_task_queue_activities_per_second,
        graceful_shutdown_timeout=graceful_shutdown_timeout,
        shared_state_manager=shared_state_manager,
        debug_mode=debug_mode,
        disable_eager_activity_execution=disable_eager_activity_execution,
        on_fatal_error=on_fatal_error,
        use_worker_versioning=use_worker_versioning,
        disable_safe_workflow_eviction=disable_safe_workflow_eviction,
        deployment_config=deployment_config,
    )
