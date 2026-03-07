"""Shared test fixtures for governance tests."""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import openbox.hook_governance as hook_gov
from openbox.types import GovernanceBlockedError, WorkflowSpanBuffer


@pytest.fixture
def cleanup_governance():
    """Reset hook_governance module state after each test."""
    yield
    hook_gov._api_url = ""
    hook_gov._api_key = ""
    hook_gov._span_processor = None


def setup_governance(
    on_api_error: str = "fail_open",
    workflow_id: str = "wf-test-1",
    activity_id: str = "act-test-1",
    workflow_type: str = "TestWorkflow",
    task_queue: str = "test-queue",
) -> MagicMock:
    """Configure hook_governance with a mock span processor.

    Returns the mock processor for further configuration.
    """
    processor = MagicMock()
    processor.get_activity_context_by_trace.return_value = {
        "workflow_id": workflow_id,
        "activity_id": activity_id,
    }
    buffer = WorkflowSpanBuffer(
        workflow_id=workflow_id, run_id="run-1",
        workflow_type=workflow_type, task_queue=task_queue,
    )
    processor.get_buffer.return_value = buffer

    hook_gov.configure(
        "http://localhost:9090", "test-key", processor,
        api_timeout=5.0, on_api_error=on_api_error,
    )
    return processor


@contextmanager
def mock_httpx_client(verdict="allow", reason=None, side_effect=None):
    """Mock httpx.Client for governance API calls (sync)."""
    response = MagicMock()
    response.status_code = 200
    response_data = {"verdict": verdict}
    if reason:
        response_data["reason"] = reason
    response.json.return_value = response_data

    mock_instance = MagicMock()
    if side_effect:
        mock_instance.__enter__ = MagicMock(side_effect=side_effect)
    else:
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.post.return_value = response
    mock_instance.__exit__ = MagicMock(return_value=False)

    with patch("openbox.hook_governance.httpx.Client", return_value=mock_instance):
        yield mock_instance


@contextmanager
def mock_httpx_async_client(verdict="allow", reason=None, side_effect=None):
    """Mock httpx.AsyncClient for async governance API calls."""
    response = MagicMock()
    response.status_code = 200
    response_data = {"verdict": verdict}
    if reason:
        response_data["reason"] = reason
    response.json.return_value = response_data

    mock_instance = MagicMock()
    if side_effect:
        mock_instance.__aenter__ = AsyncMock(side_effect=side_effect)
    else:
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.post = AsyncMock(return_value=response)
    mock_instance.__aexit__ = AsyncMock(return_value=None)

    with patch("openbox.hook_governance.httpx.AsyncClient", return_value=mock_instance):
        yield mock_instance
