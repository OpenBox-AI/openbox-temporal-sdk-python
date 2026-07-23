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
from typing import Any, Awaitable, Callable, Dict, Optional, Sequence, Type

from temporalio.client import Client
from temporalio.worker import Interceptor, Worker

from .config import GovernanceConfig
from .config import initialize as validate_api_key

logger = logging.getLogger(__name__)


def create_openbox_worker(
    client: Client,
    task_queue: str,
    *,
    workflows: Sequence[Type] = (),
    activities: Sequence[Callable] = (),
    openbox_url: str,
    openbox_api_key: str,
    agent_did: Optional[str] = None,
    agent_private_key: Optional[str] = None,
    governance_timeout: float = 30.0,
    governance_policy: str = "fail_open",
    send_start_event: bool = True,
    send_activity_start_event: bool = True,
    skip_workflow_types: Optional[set] = None,
    skip_activity_types: Optional[set] = None,
    skip_signals: Optional[set] = None,
    hitl_enabled: bool = True,
    max_patch_restarts: int = 3,
    instrument_databases: bool = True,
    db_libraries: Optional[set] = None,
    sqlalchemy_engine: Optional[Any] = None,
    instrument_file_io: bool = True,
    enable_trace_propagation: bool = True,
    activity_executor: Optional[Executor] = None,
    workflow_task_executor: Optional[ThreadPoolExecutor] = None,
    interceptors: Sequence[Interceptor] = (),
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
        governance_timeout: Timeout for governance API calls (default: 30.0s)
        governance_policy: "fail_open" or "fail_closed" (default: "fail_open")
        send_start_event: Send WorkflowStarted events (default: True)
        send_activity_start_event: Send ActivityStarted events (default: True)
        skip_workflow_types: Workflow types to skip governance
        skip_activity_types: Activity types to skip governance
        skip_signals: Signal names to skip governance

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
        instrument_file_io: Instrument file I/O operations (default: True)

        # Standard Worker options (passed through to Worker)
        activity_executor: Executor for activities
        interceptors: Additional interceptors (OpenBox interceptors added automatically)
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
    logger.info("Initializing OpenBox SDK with URL: %s", openbox_url)

    from .activities import set_temporal_client

    set_temporal_client(client)

    validate_api_key(
        api_url=openbox_url,
        api_key=openbox_api_key,
        governance_timeout=governance_timeout,
        agent_did=agent_did,
        agent_private_key=agent_private_key,
    )

    from .config import get_global_config

    _signer = get_global_config().get_signer()

    from .governance_state import TemporalGovernanceState

    state = TemporalGovernanceState()

    from .core_adapter import create_core_runtime

    runtime = create_core_runtime(
        api_url=openbox_url,
        api_key=openbox_api_key,
        state=state,
        timeout_seconds=governance_timeout,
        on_api_error=governance_policy,
        agent_did=agent_did,
        agent_private_key=agent_private_key,
        hitl_enabled=hitl_enabled,
        skip_workflow_types=skip_workflow_types or set(),
        skip_activity_types=skip_activity_types or {"send_governance_event"},
        skip_signals=skip_signals or set(),
        send_start_event=send_start_event,
        send_activity_start_event=send_activity_start_event,
        instrument_databases=instrument_databases,
        instrument_file_io=instrument_file_io,
    )
    runtime.install_instrumentation()

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
    from .client import GovernanceClient
    from .workflow_interceptor import GovernanceInterceptor

    workflow_interceptor = GovernanceInterceptor(
        api_url=openbox_url,
        api_key=openbox_api_key,
        state=state,
        config=config,
    )

    governance_client = GovernanceClient(
        api_url=openbox_url,
        api_key=openbox_api_key,
        timeout=governance_timeout,
        on_api_error=governance_policy,
        agent_did=agent_did,
        signer=_signer,
    )

    activity_interceptor = ActivityGovernanceInterceptor(
        api_url=openbox_url,
        api_key=openbox_api_key,
        state=state,
        config=config,
        client=governance_client,
    )

    from .activities import build_governance_activities

    governance_activities = build_governance_activities(
        api_url=openbox_url,
        api_key=openbox_api_key,
        agent_did=agent_did,
        signer=_signer,
    )

    all_interceptors: list = [workflow_interceptor, activity_interceptor, *interceptors]

    if enable_trace_propagation:
        from temporalio.contrib.opentelemetry import TracingInterceptor

        all_interceptors.append(TracingInterceptor())

    all_activities = [*activities, governance_activities.send_governance_event]

    logger.info(
        "OpenBox SDK initialized: policy=%s timeout=%ss db=%s file=%s hitl=%s "
        "instrumentation=openbox_core events=WorkflowStarted,WorkflowCompleted,"
        "WorkflowFailed,SignalReceived,ActivityStarted,ActivityCompleted",
        governance_policy,
        governance_timeout,
        "enabled" if instrument_databases else "disabled",
        "enabled" if instrument_file_io else "disabled",
        "enabled" if hitl_enabled else "disabled",
    )

    return Worker(
        client,
        task_queue=task_queue,
        workflows=workflows,
        activities=all_activities,
        activity_executor=activity_executor,
        workflow_task_executor=workflow_task_executor,
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
    )
