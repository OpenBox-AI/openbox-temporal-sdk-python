"""
Comprehensive pytest tests for the OpenBox SDK worker module.

Tests cover:
- create_openbox_worker() with OpenBox config
- Parameter passthrough to Worker
- Configuration options for governance

Bootstrap model under test (base-SDK instrumentation):
  create_openbox_worker() validates the key, creates a TemporalGovernanceState,
  builds+owns an openbox_core runtime via create_core_runtime(...), installs
  instrumentation, then wires both interceptors with that shared state.
"""

import functools
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from unittest.mock import MagicMock, Mock, patch

from openbox.governance_state import TemporalGovernanceState

# Shared patch stack
#
# create_openbox_worker() imports these names differently:
#   - Worker / validate_api_key / GovernanceConfig are imported at module top,
#     so they are patched on openbox.worker.
#   - create_core_runtime, the interceptors, and build_governance_activities are
#     imported inside the function from their source modules, so they are patched
#     at the source module.
#
# create_core_runtime is ALWAYS patched: the real one builds an openbox_core
# runtime and install_instrumentation() patches HTTP/DB/file globally + can hit
# the network. The mock's .install_instrumentation is a MagicMock so tests can
# assert it was invoked exactly once.

_PATCH_TARGETS = [
    ("mock_worker_class", "openbox.worker.Worker"),
    ("mock_validate_api_key", "openbox.worker.validate_api_key"),
    ("mock_governance_config", "openbox.worker.GovernanceConfig"),
    ("mock_create_core_runtime", "openbox.core_adapter.create_core_runtime"),
    (
        "mock_governance_interceptor",
        "openbox.workflow_interceptor.GovernanceInterceptor",
    ),
    (
        "mock_activity_interceptor",
        "openbox.activity_interceptor.ActivityGovernanceInterceptor",
    ),
    ("mock_build_activities", "openbox.activities.build_governance_activities"),
]


def with_worker_patches(func):
    """Apply the standard worker patch stack and inject mocks as kwargs.

    Injects one keyword argument per entry in _PATCH_TARGETS (by name), plus a
    fresh MagicMock runtime returned by create_core_runtime whose
    .install_instrumentation is assertable.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with_stack = []
        try:
            mocks = {}
            for attr, target in _PATCH_TARGETS:
                p = patch(target)
                with_stack.append(p)
                mocks[attr] = p.start()

            # create_core_runtime returns a runtime the worker installs + owns.
            runtime = MagicMock(name="core_runtime")
            runtime.install_instrumentation = MagicMock(name="install_instrumentation")
            mocks["mock_create_core_runtime"].return_value = runtime
            mocks["mock_runtime"] = runtime

            # The worker registers governance_activities.send_governance_event by
            # object identity; the "is the governance activity registered?" checks
            # match on __name__, so the default mock must model that attribute the
            # way the real bound method carries it. Tests needing a distinct
            # sentinel override .send_governance_event explicitly.
            mocks[
                "mock_build_activities"
            ].return_value.send_governance_event.__name__ = "send_governance_event"

            return func(self, *args, **{**mocks, **kwargs})
        finally:
            for p in reversed(with_stack):
                p.stop()

    return wrapper


class TestCreateOpenboxWorkerWithConfig:
    """Test create_openbox_worker() with OpenBox configuration."""

    @with_worker_patches
    def test_validates_api_key(self, **m):
        """Validates API key (initialize) with the provided credentials + timeout."""
        from openbox.worker import create_openbox_worker

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            governance_timeout=45.0,
        )

        m["mock_validate_api_key"].assert_called_once_with(
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            governance_timeout=45.0,
            agent_did=None,
            agent_private_key=None,
        )

    @with_worker_patches
    def test_builds_and_installs_core_runtime(self, **m):
        """Builds an openbox_core runtime and installs instrumentation exactly once.

        Instrumentation setup flows through create_core_runtime and
        runtime.install_instrumentation().
        """
        from openbox.worker import create_openbox_worker

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            governance_timeout=30.0,
            governance_policy="fail_open",
            instrument_databases=True,
            instrument_file_io=True,
        )

        m["mock_create_core_runtime"].assert_called_once()
        call_kwargs = m["mock_create_core_runtime"].call_args.kwargs
        assert call_kwargs["api_url"] == "http://localhost:8086"
        assert call_kwargs["api_key"] == "obx_test_key123"
        assert call_kwargs["timeout_seconds"] == 30.0
        assert call_kwargs["on_api_error"] == "fail_open"
        assert call_kwargs["instrument_databases"] is True
        assert call_kwargs["instrument_file_io"] is True
        # The runtime lives for the worker process; install must run once.
        m["mock_runtime"].install_instrumentation.assert_called_once_with()

    @with_worker_patches
    def test_core_runtime_receives_shared_state(self, **m):
        """create_core_runtime and both interceptors receive the SAME state object."""
        from openbox.worker import create_openbox_worker

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        runtime_state = m["mock_create_core_runtime"].call_args.kwargs["state"]
        assert isinstance(runtime_state, TemporalGovernanceState)

        wf_state = m["mock_governance_interceptor"].call_args.kwargs["state"]
        act_state = m["mock_activity_interceptor"].call_args.kwargs["state"]
        # One state instance shared across the runtime and both interceptors.
        assert wf_state is runtime_state
        assert act_state is runtime_state

    @with_worker_patches
    def test_creates_governance_config_with_correct_values(self, **m):
        """Creates GovernanceConfig with correct values."""
        from openbox.worker import create_openbox_worker

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            governance_timeout=60.0,
            governance_policy="fail_closed",
            send_start_event=False,
            send_activity_start_event=False,
            skip_workflow_types={"WorkflowA"},
            skip_activity_types={"activity_a", "send_governance_event"},
            skip_signals={"signal_a"},
            hitl_enabled=False,
        )

        m["mock_governance_config"].assert_called_once_with(
            on_api_error="fail_closed",
            api_timeout=60.0,
            send_start_event=False,
            send_activity_start_event=False,
            skip_workflow_types={"WorkflowA"},
            skip_activity_types={"activity_a", "send_governance_event"},
            skip_signals={"signal_a"},
            hitl_enabled=False,
            max_retryable_block_restarts=3,
        )

    @with_worker_patches
    def test_max_retryable_block_restarts_passed_to_config(self, **m):
        """A custom restart budget reaches GovernanceConfig."""
        from openbox.worker import create_openbox_worker

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            max_retryable_block_restarts=7,
        )

        assert (
            m["mock_governance_config"].call_args.kwargs["max_retryable_block_restarts"]
            == 7
        )

    @with_worker_patches
    def test_creates_governance_interceptor_with_correct_args(self, **m):
        """Creates GovernanceInterceptor with api_url/api_key/state/config."""
        from openbox.worker import create_openbox_worker

        mock_config = Mock()
        m["mock_governance_config"].return_value = mock_config

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086/",  # With trailing slash
            openbox_api_key="obx_test_key123",
        )

        m["mock_governance_interceptor"].assert_called_once()
        call_kwargs = m["mock_governance_interceptor"].call_args.kwargs
        assert call_kwargs["api_url"] == "http://localhost:8086/"
        assert call_kwargs["api_key"] == "obx_test_key123"
        assert call_kwargs["config"] == mock_config
        # span_processor is gone — the interceptor is now state-backed.
        assert "span_processor" not in call_kwargs
        assert isinstance(call_kwargs["state"], TemporalGovernanceState)

    @with_worker_patches
    def test_creates_activity_interceptor_with_correct_args(self, **m):
        """Creates ActivityGovernanceInterceptor with api_url/api_key/state/config/client."""
        from openbox.worker import create_openbox_worker

        mock_config = Mock()
        m["mock_governance_config"].return_value = mock_config

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        call_kwargs = m["mock_activity_interceptor"].call_args.kwargs
        assert call_kwargs["api_url"] == "http://localhost:8086"
        assert call_kwargs["api_key"] == "obx_test_key123"
        assert call_kwargs["config"] == mock_config
        assert "span_processor" not in call_kwargs
        assert isinstance(call_kwargs["state"], TemporalGovernanceState)
        assert "client" in call_kwargs  # GovernanceClient injected by worker

    @with_worker_patches
    def test_adds_send_governance_event_to_activities(self, **m):
        """The class-based send_governance_event method is registered on the worker."""
        from openbox.worker import create_openbox_worker

        sentinel_method = Mock(name="send_governance_event_method")
        m["mock_build_activities"].return_value.send_governance_event = sentinel_method

        def my_activity():
            pass

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            activities=[my_activity],
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            enable_trace_propagation=False,
        )

        m["mock_worker_class"].assert_called_once()
        call_kwargs = m["mock_worker_class"].call_args[1]
        assert my_activity in call_kwargs["activities"]
        assert sentinel_method in call_kwargs["activities"]
        # Credentials must be captured on the activity instance, not flowed via input
        m["mock_build_activities"].assert_called_once_with(
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            agent_did=None,
            signer=None,
        )

    @with_worker_patches
    def test_interceptors_are_prepended(self, **m):
        """OpenBox interceptors are prepended (first) in interceptor list."""
        from openbox.worker import create_openbox_worker

        mock_custom_interceptor = Mock()
        mock_workflow_interceptor = Mock()
        mock_activity_interceptor_instance = Mock()
        m["mock_governance_interceptor"].return_value = mock_workflow_interceptor
        m["mock_activity_interceptor"].return_value = mock_activity_interceptor_instance

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            interceptors=[mock_custom_interceptor],
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            enable_trace_propagation=False,
        )

        m["mock_worker_class"].assert_called_once()
        call_kwargs = m["mock_worker_class"].call_args[1]
        interceptors = call_kwargs["interceptors"]
        assert interceptors[0] == mock_workflow_interceptor
        assert interceptors[1] == mock_activity_interceptor_instance
        assert interceptors[2] == mock_custom_interceptor


class TestParameterPassthrough:
    """Test that standard Worker options are passed through correctly."""

    @with_worker_patches
    def test_basic_parameters_passed_through(self, **m):
        """Basic parameters are passed through to Worker."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        class MyWorkflow:
            pass

        def my_activity():
            pass

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            workflows=[MyWorkflow],
            activities=[my_activity],
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        m["mock_worker_class"].assert_called_once()
        args, kwargs = m["mock_worker_class"].call_args
        assert args[0] == mock_client
        assert kwargs["task_queue"] == "test-queue"
        assert kwargs["workflows"] == [MyWorkflow]
        assert my_activity in kwargs["activities"]

    @with_worker_patches
    def test_executor_parameters_passed_through(self, **m):
        """Executor parameters are passed through to Worker."""
        from openbox.worker import create_openbox_worker

        mock_activity_executor = Mock()
        mock_workflow_executor = ThreadPoolExecutor(max_workers=4)

        try:
            create_openbox_worker(
                client=Mock(),
                task_queue="test-queue",
                activity_executor=mock_activity_executor,
                workflow_task_executor=mock_workflow_executor,
                openbox_url="http://localhost:8086",
                openbox_api_key="obx_test_key123",
            )

            m["mock_worker_class"].assert_called_once()
            call_kwargs = m["mock_worker_class"].call_args[1]
            assert call_kwargs["activity_executor"] == mock_activity_executor
            assert call_kwargs["workflow_task_executor"] == mock_workflow_executor
        finally:
            mock_workflow_executor.shutdown(wait=False)

    @with_worker_patches
    def test_concurrency_parameters_passed_through(self, **m):
        """Concurrency parameters are passed through to Worker."""
        from openbox.worker import create_openbox_worker

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            max_cached_workflows=500,
            max_concurrent_workflow_tasks=10,
            max_concurrent_activities=20,
            max_concurrent_local_activities=15,
            max_concurrent_workflow_task_polls=3,
            max_concurrent_activity_task_polls=3,
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        m["mock_worker_class"].assert_called_once()
        call_kwargs = m["mock_worker_class"].call_args[1]
        assert call_kwargs["max_cached_workflows"] == 500
        assert call_kwargs["max_concurrent_workflow_tasks"] == 10
        assert call_kwargs["max_concurrent_activities"] == 20
        assert call_kwargs["max_concurrent_local_activities"] == 15
        assert call_kwargs["max_concurrent_workflow_task_polls"] == 3
        assert call_kwargs["max_concurrent_activity_task_polls"] == 3

    @with_worker_patches
    def test_timeout_parameters_passed_through(self, **m):
        """Timeout parameters are passed through to Worker."""
        from openbox.worker import create_openbox_worker

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            sticky_queue_schedule_to_start_timeout=timedelta(seconds=20),
            max_heartbeat_throttle_interval=timedelta(seconds=120),
            default_heartbeat_throttle_interval=timedelta(seconds=45),
            graceful_shutdown_timeout=timedelta(seconds=30),
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        m["mock_worker_class"].assert_called_once()
        call_kwargs = m["mock_worker_class"].call_args[1]
        assert call_kwargs["sticky_queue_schedule_to_start_timeout"] == timedelta(
            seconds=20
        )
        assert call_kwargs["max_heartbeat_throttle_interval"] == timedelta(seconds=120)
        assert call_kwargs["default_heartbeat_throttle_interval"] == timedelta(
            seconds=45
        )
        assert call_kwargs["graceful_shutdown_timeout"] == timedelta(seconds=30)

    @with_worker_patches
    def test_rate_limit_parameters_passed_through(self, **m):
        """Rate limit parameters are passed through to Worker."""
        from openbox.worker import create_openbox_worker

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            max_activities_per_second=10.0,
            max_task_queue_activities_per_second=50.0,
            nonsticky_to_sticky_poll_ratio=0.3,
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        m["mock_worker_class"].assert_called_once()
        call_kwargs = m["mock_worker_class"].call_args[1]
        assert call_kwargs["max_activities_per_second"] == 10.0
        assert call_kwargs["max_task_queue_activities_per_second"] == 50.0
        assert call_kwargs["nonsticky_to_sticky_poll_ratio"] == 0.3

    @with_worker_patches
    def test_identity_and_build_parameters_passed_through(self, **m):
        """Identity and build parameters are passed through to Worker."""
        from openbox.worker import create_openbox_worker

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            build_id="v1.2.3",
            identity="worker-1",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        m["mock_worker_class"].assert_called_once()
        call_kwargs = m["mock_worker_class"].call_args[1]
        assert call_kwargs["build_id"] == "v1.2.3"
        assert call_kwargs["identity"] == "worker-1"

    @with_worker_patches
    def test_boolean_flags_passed_through(self, **m):
        """Boolean flag parameters are passed through to Worker."""
        from openbox.worker import create_openbox_worker

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            no_remote_activities=True,
            debug_mode=True,
            disable_eager_activity_execution=True,
            use_worker_versioning=True,
            disable_safe_workflow_eviction=True,
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        m["mock_worker_class"].assert_called_once()
        call_kwargs = m["mock_worker_class"].call_args[1]
        assert call_kwargs["no_remote_activities"] is True
        assert call_kwargs["debug_mode"] is True
        assert call_kwargs["disable_eager_activity_execution"] is True
        assert call_kwargs["use_worker_versioning"] is True
        assert call_kwargs["disable_safe_workflow_eviction"] is True

    @with_worker_patches
    def test_callback_parameters_passed_through(self, **m):
        """Callback parameters are passed through to Worker."""
        from openbox.worker import create_openbox_worker

        async def on_fatal_error(error):
            pass

        mock_shared_state_manager = Mock()

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            on_fatal_error=on_fatal_error,
            shared_state_manager=mock_shared_state_manager,
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        m["mock_worker_class"].assert_called_once()
        call_kwargs = m["mock_worker_class"].call_args[1]
        assert call_kwargs["on_fatal_error"] == on_fatal_error
        assert call_kwargs["shared_state_manager"] == mock_shared_state_manager

    @with_worker_patches
    def test_custom_interceptors_appended_after_openbox(self, **m):
        """Custom interceptors are appended after OpenBox interceptors."""
        from openbox.worker import create_openbox_worker

        mock_custom_interceptor_1 = Mock(name="custom1")
        mock_custom_interceptor_2 = Mock(name="custom2")
        mock_workflow_interceptor = Mock(name="workflow")
        mock_activity_interceptor_instance = Mock(name="activity")
        m["mock_governance_interceptor"].return_value = mock_workflow_interceptor
        m["mock_activity_interceptor"].return_value = mock_activity_interceptor_instance

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            interceptors=[mock_custom_interceptor_1, mock_custom_interceptor_2],
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            enable_trace_propagation=False,
        )

        m["mock_worker_class"].assert_called_once()
        call_kwargs = m["mock_worker_class"].call_args[1]
        interceptors = call_kwargs["interceptors"]

        # Order: [workflow_interceptor, activity_interceptor, custom1, custom2]
        assert len(interceptors) == 4
        assert interceptors[0] == mock_workflow_interceptor
        assert interceptors[1] == mock_activity_interceptor_instance
        assert interceptors[2] == mock_custom_interceptor_1
        assert interceptors[3] == mock_custom_interceptor_2

    @with_worker_patches
    def test_custom_activities_preserved(self, **m):
        """Custom activities are preserved when OpenBox is configured."""
        from openbox.worker import create_openbox_worker

        def activity_a():
            pass

        def activity_b():
            pass

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            activities=[activity_a, activity_b],
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        m["mock_worker_class"].assert_called_once()
        call_kwargs = m["mock_worker_class"].call_args[1]
        activities = call_kwargs["activities"]

        # Custom activities should be first, then the class-based governance method.
        assert activity_a in activities
        assert activity_b in activities
        assert any(
            getattr(a, "__name__", "") == "send_governance_event" for a in activities
        )


class TestConfigurationOptions:
    """Test configuration options for governance."""

    @with_worker_patches
    def test_governance_timeout_passed_to_config(self, **m):
        """governance_timeout is passed to GovernanceConfig."""
        from openbox.worker import create_openbox_worker

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            governance_timeout=120.0,
        )

        m["mock_governance_config"].assert_called_once()
        call_kwargs = m["mock_governance_config"].call_args[1]
        assert call_kwargs["api_timeout"] == 120.0

    @with_worker_patches
    def test_governance_policy_fail_open_passed_to_config(self, **m):
        """governance_policy='fail_open' is passed to GovernanceConfig."""
        from openbox.worker import create_openbox_worker

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            governance_policy="fail_open",
        )

        m["mock_governance_config"].assert_called_once()
        call_kwargs = m["mock_governance_config"].call_args[1]
        assert call_kwargs["on_api_error"] == "fail_open"

    @with_worker_patches
    def test_governance_policy_fail_closed_passed_to_config(self, **m):
        """governance_policy='fail_closed' is passed to GovernanceConfig."""
        from openbox.worker import create_openbox_worker

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            governance_policy="fail_closed",
        )

        m["mock_governance_config"].assert_called_once()
        call_kwargs = m["mock_governance_config"].call_args[1]
        assert call_kwargs["on_api_error"] == "fail_closed"

    @with_worker_patches
    def test_send_start_event_passed_to_config(self, **m):
        """send_start_event is passed to GovernanceConfig."""
        from openbox.worker import create_openbox_worker

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            send_start_event=False,
        )

        m["mock_governance_config"].assert_called_once()
        call_kwargs = m["mock_governance_config"].call_args[1]
        assert call_kwargs["send_start_event"] is False

    @with_worker_patches
    def test_send_activity_start_event_passed_to_config(self, **m):
        """send_activity_start_event is passed to GovernanceConfig."""
        from openbox.worker import create_openbox_worker

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            send_activity_start_event=False,
        )

        m["mock_governance_config"].assert_called_once()
        call_kwargs = m["mock_governance_config"].call_args[1]
        assert call_kwargs["send_activity_start_event"] is False

    @with_worker_patches
    def test_skip_workflow_types_passed_to_config(self, **m):
        """skip_workflow_types is passed to GovernanceConfig."""
        from openbox.worker import create_openbox_worker

        skip_types = {"WorkflowA", "WorkflowB"}

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            skip_workflow_types=skip_types,
        )

        m["mock_governance_config"].assert_called_once()
        call_kwargs = m["mock_governance_config"].call_args[1]
        assert call_kwargs["skip_workflow_types"] == skip_types

    @with_worker_patches
    def test_skip_workflow_types_defaults_to_empty_set(self, **m):
        """skip_workflow_types defaults to empty set when None."""
        from openbox.worker import create_openbox_worker

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            skip_workflow_types=None,
        )

        m["mock_governance_config"].assert_called_once()
        call_kwargs = m["mock_governance_config"].call_args[1]
        assert call_kwargs["skip_workflow_types"] == set()

    @with_worker_patches
    def test_skip_activity_types_passed_to_config(self, **m):
        """skip_activity_types is passed to GovernanceConfig."""
        from openbox.worker import create_openbox_worker

        skip_types = {"activity_a", "activity_b"}

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            skip_activity_types=skip_types,
        )

        m["mock_governance_config"].assert_called_once()
        call_kwargs = m["mock_governance_config"].call_args[1]
        assert call_kwargs["skip_activity_types"] == skip_types

    @with_worker_patches
    def test_skip_activity_types_default_includes_send_governance_event(self, **m):
        """skip_activity_types defaults to {'send_governance_event'} when None."""
        from openbox.worker import create_openbox_worker

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            skip_activity_types=None,
        )

        m["mock_governance_config"].assert_called_once()
        call_kwargs = m["mock_governance_config"].call_args[1]
        assert "send_governance_event" in call_kwargs["skip_activity_types"]

    @with_worker_patches
    def test_skip_signals_passed_to_config(self, **m):
        """skip_signals is passed to GovernanceConfig."""
        from openbox.worker import create_openbox_worker

        skip_signals = {"signal_a", "signal_b"}

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            skip_signals=skip_signals,
        )

        m["mock_governance_config"].assert_called_once()
        call_kwargs = m["mock_governance_config"].call_args[1]
        assert call_kwargs["skip_signals"] == skip_signals

    @with_worker_patches
    def test_skip_signals_defaults_to_empty_set(self, **m):
        """skip_signals defaults to empty set when None."""
        from openbox.worker import create_openbox_worker

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            skip_signals=None,
        )

        m["mock_governance_config"].assert_called_once()
        call_kwargs = m["mock_governance_config"].call_args[1]
        assert call_kwargs["skip_signals"] == set()

    @with_worker_patches
    def test_hitl_enabled_passed_to_config(self, **m):
        """hitl_enabled is passed to GovernanceConfig."""
        from openbox.worker import create_openbox_worker

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            hitl_enabled=False,
        )

        m["mock_governance_config"].assert_called_once()
        call_kwargs = m["mock_governance_config"].call_args[1]
        assert call_kwargs["hitl_enabled"] is False

    @with_worker_patches
    def test_hitl_enabled_default_is_true(self, **m):
        """hitl_enabled defaults to True."""
        from openbox.worker import create_openbox_worker

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        m["mock_governance_config"].assert_called_once()
        call_kwargs = m["mock_governance_config"].call_args[1]
        assert call_kwargs["hitl_enabled"] is True

    @with_worker_patches
    def test_instrument_databases_passed_to_core_runtime(self, **m):
        """instrument_databases is forwarded to create_core_runtime.

        DB instrumentation configuration is owned by create_core_runtime.
        """
        from openbox.worker import create_openbox_worker

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            instrument_databases=False,
        )

        m["mock_create_core_runtime"].assert_called_once()
        call_kwargs = m["mock_create_core_runtime"].call_args.kwargs
        assert call_kwargs["instrument_databases"] is False

    @with_worker_patches
    def test_instrument_file_io_passed_to_core_runtime(self, **m):
        """instrument_file_io is forwarded to create_core_runtime."""
        from openbox.worker import create_openbox_worker

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            instrument_file_io=False,
        )

        m["mock_create_core_runtime"].assert_called_once()
        call_kwargs = m["mock_create_core_runtime"].call_args.kwargs
        assert call_kwargs["instrument_file_io"] is False

    @with_worker_patches
    def test_db_libraries_accepted_as_noop(self, **m):
        """db_libraries is still accepted (API-compat) but no longer flows anywhere.

        The base runtime installs every available DB instrumentor best-effort, so
        db_libraries is a no-op: passing it must not break worker construction and
        must not appear in the create_core_runtime call.
        """
        from openbox.worker import create_openbox_worker

        result = create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            db_libraries={"psycopg2", "asyncpg", "redis"},
        )

        assert result is m["mock_worker_class"].return_value
        assert "db_libraries" not in m["mock_create_core_runtime"].call_args.kwargs

    @with_worker_patches
    def test_sqlalchemy_engine_accepted_as_noop(self, **m):
        """sqlalchemy_engine is still accepted (API-compat) but no longer flows anywhere.

        SQLAlchemy is governed via a global Engine listener in the base runtime, so
        passing an engine must not break construction and must not reach
        create_core_runtime.
        """
        from openbox.worker import create_openbox_worker

        result = create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            sqlalchemy_engine=Mock(),
        )

        assert result is m["mock_worker_class"].return_value
        assert "sqlalchemy_engine" not in m["mock_create_core_runtime"].call_args.kwargs


class TestLogOutput:
    """Test initialization status logging."""

    @with_worker_patches
    @patch("builtins.print")
    def test_logs_initialization_messages(self, mock_print, caplog, **m):
        """Initialization status is logged via the module logger (not print())."""
        import logging

        from openbox.worker import create_openbox_worker

        with caplog.at_level(logging.INFO, logger="openbox.worker"):
            create_openbox_worker(
                client=Mock(),
                task_queue="test-queue",
                openbox_url="http://localhost:8086",
                openbox_api_key="obx_test_key123",
                governance_policy="fail_closed",
                governance_timeout=45.0,
                instrument_databases=True,
                instrument_file_io=True,
                hitl_enabled=True,
                enable_trace_propagation=False,
            )

        # Must NOT use print() — library code should emit to logger
        mock_print.assert_not_called()
        log_text = "\n".join(record.getMessage() for record in caplog.records)
        assert "Initializing OpenBox SDK with URL: http://localhost:8086" in log_text
        assert "OpenBox SDK initialized" in log_text
        assert "policy=fail_closed" in log_text
        assert "timeout=45.0s" in log_text
        assert "db=enabled" in log_text
        assert "file=enabled" in log_text
        assert "hitl=enabled" in log_text

    @with_worker_patches
    @patch("builtins.print")
    def test_logs_disabled_status_messages(self, mock_print, caplog, **m):
        """Disabled-status values surface in the logger output."""
        import logging

        from openbox.worker import create_openbox_worker

        with caplog.at_level(logging.INFO, logger="openbox.worker"):
            create_openbox_worker(
                client=Mock(),
                task_queue="test-queue",
                openbox_url="http://localhost:8086",
                openbox_api_key="obx_test_key123",
                instrument_databases=False,
                instrument_file_io=False,
                hitl_enabled=False,
                enable_trace_propagation=False,
            )

        mock_print.assert_not_called()
        log_text = "\n".join(record.getMessage() for record in caplog.records)
        assert "db=disabled" in log_text
        assert "file=disabled" in log_text
        assert "hitl=disabled" in log_text


class TestReturnValue:
    """Test return value of create_openbox_worker()."""

    @with_worker_patches
    def test_returns_worker_instance(self, **m):
        """Returns the Worker instance built by the factory."""
        from openbox.worker import create_openbox_worker

        mock_worker = Mock()
        m["mock_worker_class"].return_value = mock_worker

        result = create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        assert result == mock_worker


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @with_worker_patches
    def test_empty_workflows_and_activities(self, **m):
        """With empty workflows and activities lists."""
        from openbox.worker import create_openbox_worker

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            workflows=[],
            activities=[],
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        m["mock_worker_class"].assert_called_once()
        call_kwargs = m["mock_worker_class"].call_args[1]
        assert call_kwargs["workflows"] == []
        # Activities will include the class-based send_governance_event method.
        assert any(
            getattr(a, "__name__", "") == "send_governance_event"
            for a in call_kwargs["activities"]
        )

    @with_worker_patches
    def test_default_parameter_values(self, **m):
        """Default parameter values are passed through."""
        from openbox.worker import create_openbox_worker

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        m["mock_worker_class"].assert_called_once()
        call_kwargs = m["mock_worker_class"].call_args[1]

        # Check default values
        assert call_kwargs["max_cached_workflows"] == 1000
        assert call_kwargs["max_concurrent_workflow_task_polls"] == 5
        assert call_kwargs["nonsticky_to_sticky_poll_ratio"] == 0.2
        assert call_kwargs["max_concurrent_activity_task_polls"] == 5
        assert call_kwargs["no_remote_activities"] is False
        assert call_kwargs["sticky_queue_schedule_to_start_timeout"] == timedelta(
            seconds=10
        )
        assert call_kwargs["max_heartbeat_throttle_interval"] == timedelta(seconds=60)
        assert call_kwargs["default_heartbeat_throttle_interval"] == timedelta(
            seconds=30
        )
        assert call_kwargs["graceful_shutdown_timeout"] == timedelta()
        assert call_kwargs["debug_mode"] is False
        assert call_kwargs["disable_eager_activity_execution"] is False
        assert call_kwargs["use_worker_versioning"] is False
        assert call_kwargs["disable_safe_workflow_eviction"] is False

    @with_worker_patches
    def test_url_with_trailing_slash(self, **m):
        """URL with trailing slash is passed as-is to components."""
        from openbox.worker import create_openbox_worker

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086/",
            openbox_api_key="obx_test_key123",
        )

        # Verify URL is passed as-is to components
        # (URL normalization happens in the interceptors)
        m["mock_governance_interceptor"].assert_called_once()
        call_kwargs = m["mock_governance_interceptor"].call_args[1]
        assert call_kwargs["api_url"] == "http://localhost:8086/"

    @with_worker_patches
    def test_large_timeout_value(self, **m):
        """Large timeout value is handled correctly."""
        from openbox.worker import create_openbox_worker

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            governance_timeout=3600.0,  # 1 hour
        )

        m["mock_validate_api_key"].assert_called_once()
        call_kwargs = m["mock_validate_api_key"].call_args[1]
        assert call_kwargs["governance_timeout"] == 3600.0

    @with_worker_patches
    def test_small_timeout_value(self, **m):
        """Small timeout value is handled correctly."""
        from openbox.worker import create_openbox_worker

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            governance_timeout=0.5,  # 500ms
        )

        m["mock_validate_api_key"].assert_called_once()
        call_kwargs = m["mock_validate_api_key"].call_args[1]
        assert call_kwargs["governance_timeout"] == 0.5

    @with_worker_patches
    def test_many_custom_interceptors(self, **m):
        """With many custom interceptors."""
        from openbox.worker import create_openbox_worker

        custom_interceptors = [Mock(name=f"interceptor_{i}") for i in range(10)]

        create_openbox_worker(
            client=Mock(),
            task_queue="test-queue",
            interceptors=custom_interceptors,
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            enable_trace_propagation=False,
        )

        m["mock_worker_class"].assert_called_once()
        call_kwargs = m["mock_worker_class"].call_args[1]
        interceptors = call_kwargs["interceptors"]

        # 2 OpenBox interceptors + 10 custom = 12 total
        assert len(interceptors) == 12
        # First 2 are OpenBox interceptors
        assert interceptors[0] == m["mock_governance_interceptor"].return_value
        assert interceptors[1] == m["mock_activity_interceptor"].return_value
        # Rest are custom interceptors in order
        for i, interceptor in enumerate(custom_interceptors):
            assert interceptors[i + 2] == interceptor
