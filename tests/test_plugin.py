# tests/test_plugin.py
"""
Unit tests for OpenBoxPlugin(SimplePlugin).

Tests the plugin class in isolation with mocked dependencies.
Mirrors test patterns from test_worker.py.
"""

import dataclasses
import pytest
from unittest.mock import Mock, MagicMock, patch, call

from temporalio.plugin import SimplePlugin
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner


# All tests mock external calls — no real API or OTel needed.
PATCH_BASE = "openbox.plugin"


def _make_plugin(**overrides):
    """Create OpenBoxPlugin with all heavy dependencies mocked."""
    defaults = dict(
        openbox_url="http://localhost:8086",
        openbox_api_key="obx_test_key_123",
    )
    defaults.update(overrides)

    with (
        patch(f"{PATCH_BASE}.validate_api_key") as mock_validate,
        patch(f"{PATCH_BASE}.WorkflowSpanProcessor") as mock_sp_cls,
        patch("openbox.otel_setup.setup_opentelemetry_for_governance") as mock_otel,
        patch("openbox.workflow_interceptor.GovernanceInterceptor") as mock_wi,
        patch(
            "openbox.activity_interceptor.ActivityGovernanceInterceptor"
        ) as mock_ai,
        patch(f"{PATCH_BASE}.GovernanceClient") as mock_gc,
    ):
        from openbox.plugin import OpenBoxPlugin

        plugin = OpenBoxPlugin(**defaults)
        mocks = {
            "validate_api_key": mock_validate,
            "span_processor_cls": mock_sp_cls,
            "setup_otel": mock_otel,
            "workflow_interceptor": mock_wi,
            "activity_interceptor": mock_ai,
            "governance_client": mock_gc,
        }
    return plugin, mocks


# ═══════════════════════════════════════════════════════════════════════════════
# Construction Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPluginInit:
    """Test OpenBoxPlugin constructor."""

    def test_validates_api_key(self):
        plugin, mocks = _make_plugin(governance_timeout=45.0)
        mocks["validate_api_key"].assert_called_once_with(
            api_url="http://localhost:8086",
            api_key="obx_test_key_123",
            governance_timeout=45.0,
        )

    def test_creates_span_processor(self):
        plugin, mocks = _make_plugin()
        mocks["span_processor_cls"].assert_called_once_with(
            ignored_url_prefixes=["http://localhost:8086"]
        )

    def test_sets_up_otel(self):
        plugin, mocks = _make_plugin(
            instrument_databases=False,
            instrument_file_io=False,
            governance_timeout=15.0,
            governance_policy="fail_closed",
        )
        mocks["setup_otel"].assert_called_once()
        kw = mocks["setup_otel"].call_args
        assert kw[1]["instrument_databases"] is False
        assert kw[1]["instrument_file_io"] is False
        assert kw[1]["api_timeout"] == 15.0
        assert kw[1]["on_api_error"] == "fail_closed"

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
            patch(f"{PATCH_BASE}.WorkflowSpanProcessor"),
            patch("openbox.otel_setup.setup_opentelemetry_for_governance"),
            patch("openbox.workflow_interceptor.GovernanceInterceptor") as mock_wi,
            patch("openbox.activity_interceptor.ActivityGovernanceInterceptor"),
            patch(f"{PATCH_BASE}.GovernanceClient"),
        ):
            from openbox.plugin import OpenBoxPlugin

            OpenBoxPlugin(
                openbox_url="http://localhost:8086",
                openbox_api_key="obx_test_key_123",
                skip_workflow_types={"InternalWorkflow"},
            )
            config_arg = mock_wi.call_args[1]["config"]
            assert "InternalWorkflow" in config_arg.skip_workflow_types

    def test_invalid_api_key_raises(self):
        """Validate that bad key format raises OpenBoxAuthError."""
        from openbox.errors import OpenBoxAuthError

        with pytest.raises(OpenBoxAuthError):
            # No mocking of validate_api_key — let real validation run
            with (
                patch(f"{PATCH_BASE}.WorkflowSpanProcessor"),
                patch("openbox.otel_setup.setup_opentelemetry_for_governance"),
                patch("openbox.workflow_interceptor.GovernanceInterceptor"),
                patch("openbox.activity_interceptor.ActivityGovernanceInterceptor"),
                patch(f"{PATCH_BASE}.GovernanceClient"),
            ):
                from openbox.plugin import OpenBoxPlugin

                OpenBoxPlugin(
                    openbox_url="http://localhost:8086",
                    openbox_api_key="bad_key_format",
                )


# ═══════════════════════════════════════════════════════════════════════════════
# configure_worker Tests
# ═══════════════════════════════════════════════════════════════════════════════


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
        mock_client = Mock()
        mock_client.config.return_value = {}
        config = {"client": mock_client, "task_queue": "q"}
        # Temporarily set config client to None after creating valid config
        config["client"] = None

        with patch("openbox.activities.set_temporal_client") as mock_set:
            # SimplePlugin.configure_worker needs client, so we patch super
            with patch.object(SimplePlugin, "configure_worker", return_value=config):
                plugin.configure_worker(config)
            mock_set.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# Sandbox Passthrough Tests
# ═══════════════════════════════════════════════════════════════════════════════


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
