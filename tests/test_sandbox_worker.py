from __future__ import annotations

from dataclasses import replace
from typing import Any
from unittest.mock import patch

import pytest
from openbox_sandbox import SandboxCommandDefinition
from openbox_sandbox.registry import (
    governed_command_registry as sandbox_command_registry,
)

from openbox.governed_command_activity import governed_command_activity
from openbox.sandbox.adapter import TemporalSandboxConfigurationError
from openbox.sandbox.interceptor import GovernedCommandInterceptor
from openbox.sandbox.plugin import OpenBoxSandboxPlugin
from openbox.sandbox.worker import create_sandbox_worker

from .sandbox_test_support import sandbox_config


def test_dedicated_worker_registers_one_interceptor_and_defensive_activity() -> None:
    config, _ = sandbox_config()
    with patch("openbox.sandbox.worker.Worker") as worker_type:
        result = create_sandbox_worker(
            object(),  # type: ignore[arg-type]
            "sandbox-queue",
            sandbox=config,
        )
    assert result is worker_type.return_value
    kwargs = worker_type.call_args.kwargs
    assert isinstance(kwargs["interceptors"][0], GovernedCommandInterceptor)
    assert kwargs["workflows"] == []
    assert kwargs["activities"] == [governed_command_activity]
    assert "workflow_runner" not in kwargs


@pytest.mark.parametrize(
    "option", ["activities", "plugins", "workflow_runner", "workflows"]
)
def test_dedicated_worker_rejects_reserved_options(option: str) -> None:
    config, _ = sandbox_config()
    worker_options: dict[str, Any] = {option: object()}
    with pytest.raises(TypeError, match="reserved Worker options"):
        create_sandbox_worker(
            object(),  # type: ignore[arg-type]
            "sandbox-queue",
            sandbox=config,
            **worker_options,
        )


def test_temporal_config_rejects_profile_drift_from_engine() -> None:
    config, _ = sandbox_config()
    other_profiles = sandbox_command_registry(
        SandboxCommandDefinition("other", "/bin/false")
    ).structured_profile_bundle()
    with pytest.raises(TemporalSandboxConfigurationError):
        replace(config, profiles=other_profiles)


def test_sandbox_plugin_registers_only_command_components() -> None:
    config, _ = sandbox_config()
    with patch("temporalio.plugin.SimplePlugin.__init__", return_value=None) as init:
        OpenBoxSandboxPlugin(config)
    kwargs = init.call_args.kwargs
    assert len(kwargs["interceptors"]) == 1
    assert isinstance(kwargs["interceptors"][0], GovernedCommandInterceptor)
    assert kwargs["activities"] == [governed_command_activity]
