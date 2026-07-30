"""
Unit tests for OpenBoxPlugin(SimplePlugin).

Tests the plugin class in isolation with mocked dependencies.
Mirrors test patterns from test_worker.py.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from temporalio.plugin import SimplePlugin
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

# All tests mock external calls — no real API or instrumentation install needed.
PATCH_BASE = "openbox.plugin"


def _make_plugin(**overrides):
    """Create OpenBoxPlugin with all heavy dependencies mocked.

    create_core_runtime() and runtime.install_instrumentation() are stubbed so
    no network or real instrumentation happens. A real TemporalGovernanceState
    is used so tests can assert the same instance flows to both interceptors.
    """
    defaults = dict(
        openbox_url="http://localhost:8086",
        openbox_api_key="obx_test_key_123",
        # Skip the real TracingInterceptor by default — many tests pass Mock
        # clients that SimplePlugin's interceptor-dedup logic can't iterate.
        enable_trace_propagation=False,
    )
    defaults.update(overrides)

    mock_runtime = MagicMock(name="core_runtime")

    with (
        patch(f"{PATCH_BASE}.validate_api_key") as mock_validate,
        patch(
            "openbox.core_adapter.create_core_runtime", return_value=mock_runtime
        ) as mock_create_runtime,
        patch("openbox.workflow_interceptor.GovernanceInterceptor") as mock_wi,
        patch("openbox.activity_interceptor.ActivityGovernanceInterceptor") as mock_ai,
        patch(f"{PATCH_BASE}.GovernanceClient") as mock_gc,
    ):
        from openbox.plugin import OpenBoxPlugin

        plugin = OpenBoxPlugin(**defaults)
        mocks = {
            "validate_api_key": mock_validate,
            "create_core_runtime": mock_create_runtime,
            "runtime": mock_runtime,
            "workflow_interceptor": mock_wi,
            "activity_interceptor": mock_ai,
            "governance_client": mock_gc,
        }
    return plugin, mocks


class TestPluginInit:
    """Test OpenBoxPlugin constructor."""

    def test_validates_api_key(self):
        plugin, mocks = _make_plugin(governance_timeout=45.0)
        mocks["validate_api_key"].assert_called_once_with(
            api_url="http://localhost:8086",
            api_key="obx_test_key_123",
            governance_timeout=45.0,
            agent_did=None,
            agent_private_key=None,
            openbox_agent_id=None,
            organization_id=None,
            deployment_id=None,
            okta_agent_id=None,
            okta_agent_key_id=None,
            okta_agent_private_key=None,
            okta_agent_algorithm=None,
            agent_proof_audience=None,
        )

    def test_builds_core_runtime_and_installs_instrumentation(self):
        """Plugin builds the base-SDK runtime and installs hook instrumentation.

        Hook governance lives in openbox_core, installed once via
        runtime.install_instrumentation().
        """
        plugin, mocks = _make_plugin()

        mocks["create_core_runtime"].assert_called_once()
        # State passed to the runtime is the plugin's TemporalGovernanceState.
        from openbox.governance_state import TemporalGovernanceState

        kw = mocks["create_core_runtime"].call_args.kwargs
        assert kw["api_url"] == "http://localhost:8086"
        assert kw["api_key"] == "obx_test_key_123"
        assert isinstance(kw["state"], TemporalGovernanceState)
        assert kw["state"] is plugin._state

        # Instrumentation installed exactly once on the returned runtime.
        mocks["runtime"].install_instrumentation.assert_called_once_with()
        assert plugin._runtime is mocks["runtime"]

    def test_core_runtime_receives_instrumentation_flags(self):
        """instrument_databases / instrument_file_io / policy / timeout flow to
        the runtime builder."""
        plugin, mocks = _make_plugin(
            instrument_databases=False,
            instrument_file_io=False,
            governance_timeout=15.0,
            governance_policy="fail_closed",
        )
        kw = mocks["create_core_runtime"].call_args.kwargs
        assert kw["instrument_databases"] is False
        assert kw["instrument_file_io"] is False
        assert kw["timeout_seconds"] == 15.0
        assert kw["on_api_error"] == "fail_closed"

    def test_shared_state_flows_to_both_interceptors(self):
        """The SAME TemporalGovernanceState is handed to both interceptors —
        the signal-verdict bridge (workflow records → activity enforces)."""
        plugin, mocks = _make_plugin()

        wi_state = mocks["workflow_interceptor"].call_args.kwargs["state"]
        ai_state = mocks["activity_interceptor"].call_args.kwargs["state"]
        assert wi_state is plugin._state
        assert ai_state is plugin._state

    def test_creates_governance_interceptor(self):
        plugin, mocks = _make_plugin()
        mocks["workflow_interceptor"].assert_called_once()

    def test_creates_activity_interceptor(self):
        plugin, mocks = _make_plugin()
        mocks["activity_interceptor"].assert_called_once()

    def test_creates_governance_client(self):
        plugin, mocks = _make_plugin(
            governance_timeout=20.0, governance_policy="fail_closed"
        )
        mocks["governance_client"].assert_called_once_with(
            api_url="http://localhost:8086",
            api_key="obx_test_key_123",
            timeout=20.0,
            on_api_error="fail_closed",
            agent_did=None,
            signer=None,
            okta_identity=None,
        )

    def test_is_simple_plugin_subclass(self):
        plugin, _ = _make_plugin()
        assert isinstance(plugin, SimplePlugin)

    def test_plugin_name(self):
        plugin, _ = _make_plugin()
        assert plugin.name() == "openbox.OpenBoxPlugin"

    def test_default_params(self):
        plugin, _ = _make_plugin()
        assert plugin._governance_policy == "fail_open"
        assert plugin._governance_timeout == 30.0
        assert plugin._instrument_databases is True
        assert plugin._instrument_file_io is True
        assert plugin._hitl_enabled is True

    def test_custom_params(self):
        plugin, _ = _make_plugin(
            governance_policy="fail_closed",
            governance_timeout=10.0,
            instrument_databases=False,
            instrument_file_io=False,
            hitl_enabled=False,
        )
        assert plugin._governance_policy == "fail_closed"
        assert plugin._governance_timeout == 10.0
        assert plugin._instrument_databases is False
        assert plugin._instrument_file_io is False
        assert plugin._hitl_enabled is False

    def test_skip_workflow_types_passed_to_config(self):
        """Verify skip_workflow_types reaches GovernanceConfig."""
        with (
            patch(f"{PATCH_BASE}.validate_api_key"),
            patch(
                "openbox.core_adapter.create_core_runtime",
                return_value=MagicMock(),
            ),
            patch("openbox.workflow_interceptor.GovernanceInterceptor") as mock_wi,
            patch("openbox.activity_interceptor.ActivityGovernanceInterceptor"),
            patch(f"{PATCH_BASE}.GovernanceClient"),
        ):
            from openbox.plugin import OpenBoxPlugin

            OpenBoxPlugin(
                openbox_url="http://localhost:8086",
                openbox_api_key="obx_test_key_123",
                skip_workflow_types={"InternalWorkflow"},
                enable_trace_propagation=False,
            )
            config_arg = mock_wi.call_args.kwargs["config"]
            assert "InternalWorkflow" in config_arg.skip_workflow_types

    def test_max_patch_restarts_passed_to_config(self):
        """Verify max_patch_restarts reaches GovernanceConfig."""
        with (
            patch(f"{PATCH_BASE}.validate_api_key"),
            patch(
                "openbox.core_adapter.create_core_runtime",
                return_value=MagicMock(),
            ),
            patch("openbox.workflow_interceptor.GovernanceInterceptor") as mock_wi,
            patch("openbox.activity_interceptor.ActivityGovernanceInterceptor"),
            patch(f"{PATCH_BASE}.GovernanceClient"),
        ):
            from openbox.plugin import OpenBoxPlugin

            OpenBoxPlugin(
                openbox_url="http://localhost:8086",
                openbox_api_key="obx_test_key_123",
                max_patch_restarts=7,
                enable_trace_propagation=False,
            )
            config_arg = mock_wi.call_args.kwargs["config"]
            assert config_arg.max_patch_restarts == 7

    def test_invalid_api_key_raises(self):
        """Validate that bad key format raises OpenBoxAuthError."""
        from openbox.errors import OpenBoxAuthError

        with pytest.raises(OpenBoxAuthError):
            # No mocking of validate_api_key — let real validation run.
            with (
                patch(
                    "openbox.core_adapter.create_core_runtime",
                    return_value=MagicMock(),
                ),
                patch("openbox.workflow_interceptor.GovernanceInterceptor"),
                patch("openbox.activity_interceptor.ActivityGovernanceInterceptor"),
                patch(f"{PATCH_BASE}.GovernanceClient"),
            ):
                from openbox.plugin import OpenBoxPlugin

                OpenBoxPlugin(
                    openbox_url="http://localhost:8086",
                    openbox_api_key="bad_key_format",
                    enable_trace_propagation=False,
                )


class TestPluginConfigureWorker:
    """Test OpenBoxPlugin.configure_worker()."""

    def test_sets_temporal_client(self):
        plugin, _ = _make_plugin()
        mock_client = Mock()
        config = {"client": mock_client, "task_queue": "q"}

        with patch("openbox.activities.set_temporal_client") as mock_set:
            plugin.configure_worker(config)
            mock_set.assert_called_once_with(mock_client)

    def test_delegates_to_super(self):
        """Verify super().configure_worker() is called (activities/interceptors appended)."""
        plugin, _ = _make_plugin()
        mock_client = Mock()
        # Minimal WorkerConfig-like dict
        config = {
            "client": mock_client,
            "task_queue": "q",
            "activities": [],
            "interceptors": [],
        }

        with patch("openbox.activities.set_temporal_client"):
            result = plugin.configure_worker(config)

        # SimplePlugin.configure_worker appends activities
        assert len(result.get("activities", [])) > 0

    def test_client_none_does_not_set(self):
        """If config client is None, set_temporal_client is not called."""
        plugin, _ = _make_plugin()
        config = {"client": None, "task_queue": "q"}

        with patch("openbox.activities.set_temporal_client") as mock_set:
            # SimplePlugin.configure_worker needs client, so we patch super
            with patch.object(SimplePlugin, "configure_worker", return_value=config):
                plugin.configure_worker(config)
            mock_set.assert_not_called()


class TestPluginWorkflowRunner:
    """Test the workflow_runner callback for sandbox passthrough."""

    def test_sandbox_passthrough_adds_opentelemetry(self):
        plugin, _ = _make_plugin()
        runner = SandboxedWorkflowRunner()
        original_modules = set(runner.restrictions.passthrough_modules)

        # The workflow_runner is a callable stored on plugin
        result = plugin.workflow_runner(runner)

        assert isinstance(result, SandboxedWorkflowRunner)
        new_modules = set(result.restrictions.passthrough_modules)
        assert "opentelemetry" in (new_modules - original_modules)

    def test_non_sandbox_runner_returned_unchanged(self):
        plugin, _ = _make_plugin()
        runner = Mock(spec=[])  # non-SandboxedWorkflowRunner

        result = plugin.workflow_runner(runner)

        assert result is runner

    def test_none_runner_returns_none(self):
        """When no runner exists (e.g. Replayer), callback returns None gracefully."""
        plugin, _ = _make_plugin()

        result = plugin.workflow_runner(None)
        assert result is None
