# tests/test_worker.py
"""
Comprehensive pytest tests for the OpenBox SDK worker module.

Tests cover:
- create_openbox_worker() without OpenBox config
- create_openbox_worker() with OpenBox config
- Parameter passthrough to Worker
- Configuration options for governance
"""

import pytest
from datetime import timedelta
from unittest.mock import Mock, MagicMock, patch, call
from concurrent.futures import ThreadPoolExecutor


# ===============================================================================
# With OpenBox Config Tests
# ===============================================================================


class TestCreateOpenboxWorkerWithConfig:
    """Test create_openbox_worker() with OpenBox configuration."""

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_validates_api_key(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test validates API key when config is provided."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            governance_timeout=45.0,
        )

        # Verify validate_api_key (initialize) was called
        mock_validate_api_key.assert_called_once_with(
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            governance_timeout=45.0,
        )

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_creates_workflow_span_processor_with_ignored_urls(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test creates WorkflowSpanProcessor with ignored URLs."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        # Verify WorkflowSpanProcessor was created with ignored URL
        mock_span_processor_class.assert_called_once_with(
            ignored_url_prefixes=["http://localhost:8086"]
        )

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_calls_setup_opentelemetry_with_correct_args(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test calls setup_opentelemetry_for_governance with correct args."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()
        mock_span_processor = Mock()
        mock_span_processor_class.return_value = mock_span_processor

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            instrument_databases=True,
            db_libraries={"psycopg2", "redis"},
            instrument_file_io=True,
        )

        # Verify setup_opentelemetry_for_governance was called correctly
        mock_setup_otel.assert_called_once_with(
            mock_span_processor,
            ignored_urls=["http://localhost:8086"],
            instrument_databases=True,
            db_libraries={"psycopg2", "redis"},
            instrument_file_io=True,
            sqlalchemy_engine=None,
        )

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_creates_governance_config_with_correct_values(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test creates GovernanceConfig with correct values."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        create_openbox_worker(
            client=mock_client,
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

        # Verify GovernanceConfig was created with correct values
        mock_governance_config.assert_called_once_with(
            on_api_error="fail_closed",
            api_timeout=60.0,
            send_start_event=False,
            send_activity_start_event=False,
            skip_workflow_types={"WorkflowA"},
            skip_activity_types={"activity_a", "send_governance_event"},
            skip_signals={"signal_a"},
            hitl_enabled=False,
        )

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_creates_governance_interceptor_with_correct_args(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test creates GovernanceInterceptor with correct args."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()
        mock_span_processor = Mock()
        mock_config = Mock()
        mock_span_processor_class.return_value = mock_span_processor
        mock_governance_config.return_value = mock_config

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086/",  # With trailing slash
            openbox_api_key="obx_test_key123",
        )

        # Verify GovernanceInterceptor was created with correct args
        mock_governance_interceptor.assert_called_once_with(
            api_url="http://localhost:8086/",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=mock_config,
        )

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_creates_activity_interceptor_with_correct_args(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test creates ActivityGovernanceInterceptor with correct args."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()
        mock_span_processor = Mock()
        mock_config = Mock()
        mock_span_processor_class.return_value = mock_span_processor
        mock_governance_config.return_value = mock_config

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        # Verify ActivityGovernanceInterceptor was created with correct args
        mock_activity_interceptor.assert_called_once_with(
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=mock_config,
        )

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_adds_send_governance_event_to_activities(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test adds send_governance_event to activities."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        def my_activity():
            pass

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            activities=[my_activity],
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        # Verify Worker was called with send_governance_event in activities
        mock_worker_class.assert_called_once()
        call_kwargs = mock_worker_class.call_args[1]
        assert my_activity in call_kwargs["activities"]
        assert mock_send_governance_event in call_kwargs["activities"]

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_interceptors_are_prepended(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test OpenBox interceptors are prepended (first) in interceptor list."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()
        mock_custom_interceptor = Mock()
        mock_workflow_interceptor = Mock()
        mock_activity_interceptor_instance = Mock()
        mock_governance_interceptor.return_value = mock_workflow_interceptor
        mock_activity_interceptor.return_value = mock_activity_interceptor_instance

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            interceptors=[mock_custom_interceptor],
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        # Verify interceptors order: [workflow_interceptor, activity_interceptor, custom]
        mock_worker_class.assert_called_once()
        call_kwargs = mock_worker_class.call_args[1]
        interceptors = call_kwargs["interceptors"]
        assert interceptors[0] == mock_workflow_interceptor
        assert interceptors[1] == mock_activity_interceptor_instance
        assert interceptors[2] == mock_custom_interceptor


# ===============================================================================
# Parameter Passthrough Tests
# ===============================================================================


class TestParameterPassthrough:
    """Test that standard Worker options are passed through correctly."""

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_basic_parameters_passed_through(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test basic parameters are passed through to Worker."""
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

        mock_worker_class.assert_called_once()
        args, kwargs = mock_worker_class.call_args
        assert args[0] == mock_client
        assert kwargs["task_queue"] == "test-queue"
        assert kwargs["workflows"] == [MyWorkflow]
        assert my_activity in kwargs["activities"]

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_executor_parameters_passed_through(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test executor parameters are passed through to Worker."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()
        mock_activity_executor = Mock()
        mock_workflow_executor = ThreadPoolExecutor(max_workers=4)

        try:
            create_openbox_worker(
                client=mock_client,
                task_queue="test-queue",
                activity_executor=mock_activity_executor,
                workflow_task_executor=mock_workflow_executor,
                openbox_url="http://localhost:8086",
                openbox_api_key="obx_test_key123",
            )

            mock_worker_class.assert_called_once()
            call_kwargs = mock_worker_class.call_args[1]
            assert call_kwargs["activity_executor"] == mock_activity_executor
            assert call_kwargs["workflow_task_executor"] == mock_workflow_executor
        finally:
            mock_workflow_executor.shutdown(wait=False)

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_concurrency_parameters_passed_through(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test concurrency parameters are passed through to Worker."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        create_openbox_worker(
            client=mock_client,
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

        mock_worker_class.assert_called_once()
        call_kwargs = mock_worker_class.call_args[1]
        assert call_kwargs["max_cached_workflows"] == 500
        assert call_kwargs["max_concurrent_workflow_tasks"] == 10
        assert call_kwargs["max_concurrent_activities"] == 20
        assert call_kwargs["max_concurrent_local_activities"] == 15
        assert call_kwargs["max_concurrent_workflow_task_polls"] == 3
        assert call_kwargs["max_concurrent_activity_task_polls"] == 3

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_timeout_parameters_passed_through(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test timeout parameters are passed through to Worker."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            sticky_queue_schedule_to_start_timeout=timedelta(seconds=20),
            max_heartbeat_throttle_interval=timedelta(seconds=120),
            default_heartbeat_throttle_interval=timedelta(seconds=45),
            graceful_shutdown_timeout=timedelta(seconds=30),
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        mock_worker_class.assert_called_once()
        call_kwargs = mock_worker_class.call_args[1]
        assert call_kwargs["sticky_queue_schedule_to_start_timeout"] == timedelta(seconds=20)
        assert call_kwargs["max_heartbeat_throttle_interval"] == timedelta(seconds=120)
        assert call_kwargs["default_heartbeat_throttle_interval"] == timedelta(seconds=45)
        assert call_kwargs["graceful_shutdown_timeout"] == timedelta(seconds=30)

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_rate_limit_parameters_passed_through(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test rate limit parameters are passed through to Worker."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            max_activities_per_second=10.0,
            max_task_queue_activities_per_second=50.0,
            nonsticky_to_sticky_poll_ratio=0.3,
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        mock_worker_class.assert_called_once()
        call_kwargs = mock_worker_class.call_args[1]
        assert call_kwargs["max_activities_per_second"] == 10.0
        assert call_kwargs["max_task_queue_activities_per_second"] == 50.0
        assert call_kwargs["nonsticky_to_sticky_poll_ratio"] == 0.3

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_identity_and_build_parameters_passed_through(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test identity and build parameters are passed through to Worker."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            build_id="v1.2.3",
            identity="worker-1",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        mock_worker_class.assert_called_once()
        call_kwargs = mock_worker_class.call_args[1]
        assert call_kwargs["build_id"] == "v1.2.3"
        assert call_kwargs["identity"] == "worker-1"

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_boolean_flags_passed_through(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test boolean flag parameters are passed through to Worker."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            no_remote_activities=True,
            debug_mode=True,
            disable_eager_activity_execution=True,
            use_worker_versioning=True,
            disable_safe_workflow_eviction=True,
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        mock_worker_class.assert_called_once()
        call_kwargs = mock_worker_class.call_args[1]
        assert call_kwargs["no_remote_activities"] is True
        assert call_kwargs["debug_mode"] is True
        assert call_kwargs["disable_eager_activity_execution"] is True
        assert call_kwargs["use_worker_versioning"] is True
        assert call_kwargs["disable_safe_workflow_eviction"] is True

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_callback_parameters_passed_through(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test callback parameters are passed through to Worker."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        async def on_fatal_error(error):
            pass

        mock_shared_state_manager = Mock()

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            on_fatal_error=on_fatal_error,
            shared_state_manager=mock_shared_state_manager,
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        mock_worker_class.assert_called_once()
        call_kwargs = mock_worker_class.call_args[1]
        assert call_kwargs["on_fatal_error"] == on_fatal_error
        assert call_kwargs["shared_state_manager"] == mock_shared_state_manager

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_custom_interceptors_appended_after_openbox(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test custom interceptors are appended after OpenBox interceptors."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()
        mock_custom_interceptor_1 = Mock(name="custom1")
        mock_custom_interceptor_2 = Mock(name="custom2")
        mock_workflow_interceptor = Mock(name="workflow")
        mock_activity_interceptor_instance = Mock(name="activity")
        mock_governance_interceptor.return_value = mock_workflow_interceptor
        mock_activity_interceptor.return_value = mock_activity_interceptor_instance

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            interceptors=[mock_custom_interceptor_1, mock_custom_interceptor_2],
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        mock_worker_class.assert_called_once()
        call_kwargs = mock_worker_class.call_args[1]
        interceptors = call_kwargs["interceptors"]

        # Order: [workflow_interceptor, activity_interceptor, custom1, custom2]
        assert len(interceptors) == 4
        assert interceptors[0] == mock_workflow_interceptor
        assert interceptors[1] == mock_activity_interceptor_instance
        assert interceptors[2] == mock_custom_interceptor_1
        assert interceptors[3] == mock_custom_interceptor_2

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_custom_activities_preserved(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test custom activities are preserved when OpenBox is configured."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        def activity_a():
            pass

        def activity_b():
            pass

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            activities=[activity_a, activity_b],
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        mock_worker_class.assert_called_once()
        call_kwargs = mock_worker_class.call_args[1]
        activities = call_kwargs["activities"]

        # Custom activities should be first, then send_governance_event
        assert activity_a in activities
        assert activity_b in activities
        assert mock_send_governance_event in activities


# ===============================================================================
# Configuration Options Tests
# ===============================================================================


class TestConfigurationOptions:
    """Test configuration options for governance."""

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_governance_timeout_passed_to_config(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test governance_timeout is passed to GovernanceConfig."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            governance_timeout=120.0,
        )

        mock_governance_config.assert_called_once()
        call_kwargs = mock_governance_config.call_args[1]
        assert call_kwargs["api_timeout"] == 120.0

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_governance_policy_fail_open_passed_to_config(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test governance_policy='fail_open' is passed to GovernanceConfig."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            governance_policy="fail_open",
        )

        mock_governance_config.assert_called_once()
        call_kwargs = mock_governance_config.call_args[1]
        assert call_kwargs["on_api_error"] == "fail_open"

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_governance_policy_fail_closed_passed_to_config(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test governance_policy='fail_closed' is passed to GovernanceConfig."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            governance_policy="fail_closed",
        )

        mock_governance_config.assert_called_once()
        call_kwargs = mock_governance_config.call_args[1]
        assert call_kwargs["on_api_error"] == "fail_closed"

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_send_start_event_passed_to_config(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test send_start_event is passed to GovernanceConfig."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        # Test with False
        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            send_start_event=False,
        )

        mock_governance_config.assert_called_once()
        call_kwargs = mock_governance_config.call_args[1]
        assert call_kwargs["send_start_event"] is False

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_send_activity_start_event_passed_to_config(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test send_activity_start_event is passed to GovernanceConfig."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            send_activity_start_event=False,
        )

        mock_governance_config.assert_called_once()
        call_kwargs = mock_governance_config.call_args[1]
        assert call_kwargs["send_activity_start_event"] is False

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_skip_workflow_types_passed_to_config(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test skip_workflow_types is passed to GovernanceConfig."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()
        skip_types = {"WorkflowA", "WorkflowB"}

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            skip_workflow_types=skip_types,
        )

        mock_governance_config.assert_called_once()
        call_kwargs = mock_governance_config.call_args[1]
        assert call_kwargs["skip_workflow_types"] == skip_types

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_skip_workflow_types_defaults_to_empty_set(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test skip_workflow_types defaults to empty set when None."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            skip_workflow_types=None,
        )

        mock_governance_config.assert_called_once()
        call_kwargs = mock_governance_config.call_args[1]
        assert call_kwargs["skip_workflow_types"] == set()

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_skip_activity_types_passed_to_config(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test skip_activity_types is passed to GovernanceConfig."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()
        skip_types = {"activity_a", "activity_b"}

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            skip_activity_types=skip_types,
        )

        mock_governance_config.assert_called_once()
        call_kwargs = mock_governance_config.call_args[1]
        assert call_kwargs["skip_activity_types"] == skip_types

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_skip_activity_types_default_includes_send_governance_event(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test skip_activity_types defaults to {'send_governance_event'} when None."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            skip_activity_types=None,
        )

        mock_governance_config.assert_called_once()
        call_kwargs = mock_governance_config.call_args[1]
        assert "send_governance_event" in call_kwargs["skip_activity_types"]

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_skip_signals_passed_to_config(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test skip_signals is passed to GovernanceConfig."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()
        skip_signals = {"signal_a", "signal_b"}

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            skip_signals=skip_signals,
        )

        mock_governance_config.assert_called_once()
        call_kwargs = mock_governance_config.call_args[1]
        assert call_kwargs["skip_signals"] == skip_signals

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_skip_signals_defaults_to_empty_set(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test skip_signals defaults to empty set when None."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            skip_signals=None,
        )

        mock_governance_config.assert_called_once()
        call_kwargs = mock_governance_config.call_args[1]
        assert call_kwargs["skip_signals"] == set()

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_hitl_enabled_passed_to_config(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test hitl_enabled is passed to GovernanceConfig."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            hitl_enabled=False,
        )

        mock_governance_config.assert_called_once()
        call_kwargs = mock_governance_config.call_args[1]
        assert call_kwargs["hitl_enabled"] is False

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_hitl_enabled_default_is_true(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test hitl_enabled defaults to True."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        mock_governance_config.assert_called_once()
        call_kwargs = mock_governance_config.call_args[1]
        assert call_kwargs["hitl_enabled"] is True

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_instrument_databases_passed_to_setup(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test instrument_databases is passed to setup_opentelemetry_for_governance."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            instrument_databases=False,
        )

        mock_setup_otel.assert_called_once()
        call_kwargs = mock_setup_otel.call_args[1]
        assert call_kwargs["instrument_databases"] is False

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_db_libraries_passed_to_setup(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test db_libraries is passed to setup_opentelemetry_for_governance."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()
        db_libs = {"psycopg2", "asyncpg", "redis"}

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            db_libraries=db_libs,
        )

        mock_setup_otel.assert_called_once()
        call_kwargs = mock_setup_otel.call_args[1]
        assert call_kwargs["db_libraries"] == db_libs

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_instrument_file_io_passed_to_setup(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test instrument_file_io is passed to setup_opentelemetry_for_governance."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            instrument_file_io=True,
        )

        mock_setup_otel.assert_called_once()
        call_kwargs = mock_setup_otel.call_args[1]
        assert call_kwargs["instrument_file_io"] is True

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_sqlalchemy_engine_passed_to_setup(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test sqlalchemy_engine is passed to setup_opentelemetry_for_governance."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()
        mock_engine = Mock()

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            sqlalchemy_engine=mock_engine,
        )

        mock_setup_otel.assert_called_once()
        call_kwargs = mock_setup_otel.call_args[1]
        assert call_kwargs["sqlalchemy_engine"] is mock_engine

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_sqlalchemy_engine_defaults_to_none(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test sqlalchemy_engine defaults to None when not provided."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        mock_setup_otel.assert_called_once()
        call_kwargs = mock_setup_otel.call_args[1]
        assert call_kwargs["sqlalchemy_engine"] is None


# ===============================================================================
# Print Output Tests
# ===============================================================================


class TestPrintOutput:
    """Test print output messages."""

    @patch("builtins.print")
    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_prints_initialization_messages(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
        mock_print,
    ):
        """Test prints initialization messages when OpenBox is configured."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            governance_policy="fail_closed",
            governance_timeout=45.0,
            instrument_databases=True,
            instrument_file_io=True,
            hitl_enabled=True,
        )

        # Verify print calls
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert "Initializing OpenBox SDK with URL: http://localhost:8086" in print_calls
        assert "OpenBox SDK initialized successfully" in print_calls
        assert "  - Governance policy: fail_closed" in print_calls
        assert "  - Governance timeout: 45.0s" in print_calls
        assert "  - Database instrumentation: enabled" in print_calls
        assert "  - File I/O instrumentation: enabled" in print_calls
        assert "  - Approval polling: enabled" in print_calls

    @patch("builtins.print")
    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_prints_disabled_status_messages(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
        mock_print,
    ):
        """Test prints disabled status when features are turned off."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            instrument_databases=False,
            instrument_file_io=False,
            hitl_enabled=False,
        )

        # Verify print calls
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert "  - Database instrumentation: disabled" in print_calls
        assert "  - File I/O instrumentation: disabled" in print_calls
        assert "  - Approval polling: disabled" in print_calls


# ===============================================================================
# Return Value Tests
# ===============================================================================


class TestReturnValue:
    """Test return value of create_openbox_worker()."""

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_returns_worker_instance(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test returns Worker instance."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()
        mock_worker = Mock()
        mock_worker_class.return_value = mock_worker

        result = create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        assert result == mock_worker

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_returns_worker_instance_with_openbox_config(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test returns Worker instance when OpenBox is configured."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()
        mock_worker = Mock()
        mock_worker_class.return_value = mock_worker

        result = create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        assert result == mock_worker


# ===============================================================================
# Edge Cases Tests
# ===============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_empty_workflows_and_activities(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test with empty workflows and activities lists."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            workflows=[],
            activities=[],
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        mock_worker_class.assert_called_once()
        call_kwargs = mock_worker_class.call_args[1]
        assert call_kwargs["workflows"] == []
        # activities will include send_governance_event
        assert mock_send_governance_event in call_kwargs["activities"]

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_default_parameter_values(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test default parameter values are passed through."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        mock_worker_class.assert_called_once()
        call_kwargs = mock_worker_class.call_args[1]

        # Check default values
        assert call_kwargs["max_cached_workflows"] == 1000
        assert call_kwargs["max_concurrent_workflow_task_polls"] == 5
        assert call_kwargs["nonsticky_to_sticky_poll_ratio"] == 0.2
        assert call_kwargs["max_concurrent_activity_task_polls"] == 5
        assert call_kwargs["no_remote_activities"] is False
        assert call_kwargs["sticky_queue_schedule_to_start_timeout"] == timedelta(seconds=10)
        assert call_kwargs["max_heartbeat_throttle_interval"] == timedelta(seconds=60)
        assert call_kwargs["default_heartbeat_throttle_interval"] == timedelta(seconds=30)
        assert call_kwargs["graceful_shutdown_timeout"] == timedelta()
        assert call_kwargs["debug_mode"] is False
        assert call_kwargs["disable_eager_activity_execution"] is False
        assert call_kwargs["use_worker_versioning"] is False
        assert call_kwargs["disable_safe_workflow_eviction"] is False

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_url_with_trailing_slash(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test URL with trailing slash is handled correctly."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086/",
            openbox_api_key="obx_test_key123",
        )

        # Verify URL is passed as-is to components
        # (URL normalization happens in the interceptors)
        mock_governance_interceptor.assert_called_once()
        call_kwargs = mock_governance_interceptor.call_args[1]
        assert call_kwargs["api_url"] == "http://localhost:8086/"

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_large_timeout_value(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test large timeout value is handled correctly."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            governance_timeout=3600.0,  # 1 hour
        )

        mock_validate_api_key.assert_called_once()
        call_kwargs = mock_validate_api_key.call_args[1]
        assert call_kwargs["governance_timeout"] == 3600.0

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_small_timeout_value(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test small timeout value is handled correctly."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
            governance_timeout=0.5,  # 500ms
        )

        mock_validate_api_key.assert_called_once()
        call_kwargs = mock_validate_api_key.call_args[1]
        assert call_kwargs["governance_timeout"] == 0.5

    @patch("openbox.worker.Worker")
    @patch("openbox.worker.validate_api_key")
    @patch("openbox.worker.WorkflowSpanProcessor")
    @patch("openbox.worker.GovernanceConfig")
    @patch("openbox.otel_setup.setup_opentelemetry_for_governance")
    @patch("openbox.workflow_interceptor.GovernanceInterceptor")
    @patch("openbox.activity_interceptor.ActivityGovernanceInterceptor")
    @patch("openbox.activities.send_governance_event")
    def test_many_custom_interceptors(
        self,
        mock_send_governance_event,
        mock_activity_interceptor,
        mock_governance_interceptor,
        mock_setup_otel,
        mock_governance_config,
        mock_span_processor_class,
        mock_validate_api_key,
        mock_worker_class,
    ):
        """Test with many custom interceptors."""
        from openbox.worker import create_openbox_worker

        mock_client = Mock()
        custom_interceptors = [Mock(name=f"interceptor_{i}") for i in range(10)]

        create_openbox_worker(
            client=mock_client,
            task_queue="test-queue",
            interceptors=custom_interceptors,
            openbox_url="http://localhost:8086",
            openbox_api_key="obx_test_key123",
        )

        mock_worker_class.assert_called_once()
        call_kwargs = mock_worker_class.call_args[1]
        interceptors = call_kwargs["interceptors"]

        # 2 OpenBox interceptors + 10 custom = 12 total
        assert len(interceptors) == 12
        # First 2 are OpenBox interceptors
        assert interceptors[0] == mock_governance_interceptor.return_value
        assert interceptors[1] == mock_activity_interceptor.return_value
        # Rest are custom interceptors in order
        for i, interceptor in enumerate(custom_interceptors):
            assert interceptors[i + 2] == interceptor
