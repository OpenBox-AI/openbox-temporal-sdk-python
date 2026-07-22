from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from datetime import timedelta
from unittest.mock import AsyncMock, patch

import pytest
from openbox_sandbox.contracts import (
    GOVERNED_COMMAND_ACTIVITY_TYPE,
    GovernedCommandActivityResult,
    GovernedCommandRequest,
)
from temporalio.workflow import ActivityCancellationType

from openbox.workflow_commands import execute_governed_command


@pytest.mark.asyncio
async def test_helper_schedules_exactly_one_attempt_with_structured_history_input() -> (
    None
):
    request = GovernedCommandRequest(
        "proof", {"mode": "safe", "count": 3, "job": "job-1"}
    )
    expected = GovernedCommandActivityResult(
        "proof", "executed_in_sandbox", 7, "not_observed", "deleted", 10, 10
    )
    execute = AsyncMock(return_value=expected)
    with patch("openbox.workflow_commands.workflow.execute_activity", execute):
        result = await execute_governed_command(
            request, start_to_close_timeout=timedelta(minutes=2)
        )
    assert result == expected
    args, kwargs = execute.call_args
    history_value = request.to_history_value()
    assert args == (GOVERNED_COMMAND_ACTIVITY_TYPE, history_value)
    assert set(history_value) == {"profile_id", "arguments"}
    assert "receipt" not in history_value
    assert kwargs["result_type"] is GovernedCommandActivityResult
    assert kwargs["retry_policy"].maximum_attempts == 1
    assert kwargs["start_to_close_timeout"] == timedelta(minutes=2)
    assert kwargs["heartbeat_timeout"] == timedelta(minutes=2)
    assert (
        kwargs["cancellation_type"]
        is ActivityCancellationType.WAIT_CANCELLATION_COMPLETED
    )
    assert "argv" not in history_value
    assert "command" not in history_value
    assert "cmd" not in history_value
    assert "code" not in history_value


@pytest.mark.asyncio
async def test_helper_rejects_nonrequest_and_nonpositive_timeout_before_schedule() -> (
    None
):
    execute = AsyncMock()
    with patch("openbox.workflow_commands.workflow.execute_activity", execute):
        with pytest.raises(TypeError):
            await execute_governed_command({"profile_id": "proof"})  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            await execute_governed_command(
                GovernedCommandRequest("proof", {}),
                start_to_close_timeout=timedelta(0),
            )
        with pytest.raises(ValueError):
            await execute_governed_command(
                GovernedCommandRequest("proof", {}),
                heartbeat_timeout=timedelta(0),
            )
    execute.assert_not_called()


def test_workflow_helper_does_not_eagerly_import_deployment_io() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import openbox.workflow_commands; "
            "forbidden=('openbox.sandbox','openbox_sandbox.engine',"
            "'openbox_sandbox.receipts','openbox_sandbox.runtime','cryptography'); "
            "assert not [name for name in sys.modules if any("
            "name == prefix or name.startswith(prefix + '.') for prefix in forbidden)]",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_workflow_helper_source_has_no_io_signing_logging_or_dispatcher_imports() -> (
    None
):
    source = inspect.getsource(__import__("openbox.workflow_commands", fromlist=["*"]))
    tree = ast.parse(source)
    imports = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    for forbidden in (
        "governed_dispatcher",
        "httpx",
        "urllib",
        "logging",
        "hashlib",
        "hmac",
        "sandbox_runtime_client",
    ):
        assert not any(
            name == forbidden or name.startswith(forbidden + ".") for name in imports
        )
    assert "TRY_CANCEL" not in source
    assert "ABANDON" not in source
