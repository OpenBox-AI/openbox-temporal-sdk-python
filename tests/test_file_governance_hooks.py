"""Tests for file I/O hook-level governance.

Verifies that file operations (open/close) trigger governance evaluations
at 'started' and 'completed' stages when instrument_file_io=True.
"""

import os
import tempfile
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

import openbox.hook_governance as hook_gov
import openbox.otel_setup as otel_setup
from openbox.types import GovernanceBlockedError, Verdict, WorkflowSpanBuffer


@pytest.fixture(autouse=True)
def cleanup_file_io():
    """Ensure file I/O instrumentation is cleaned up after each test."""
    yield
    otel_setup.uninstrument_file_io()
    hook_gov._api_url = ""
    hook_gov._api_key = ""
    hook_gov._span_processor = None
    otel_setup._span_processor = None


def _make_temp_file(content: bytes = b"test data") -> str:
    """Create a temp file using os-level calls (bypasses builtins.open)."""
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.write(fd, content)
    os.close(fd)
    return path


def _setup_governance(on_api_error: str = "fail_open") -> MagicMock:
    """Set up file I/O instrumentation with governance enabled."""
    processor = MagicMock()
    processor.store_body = MagicMock()
    processor.mark_governed = MagicMock()
    processor.get_activity_context_by_trace.return_value = {
        "workflow_id": "wf-file-1",
        "activity_id": "act-file-1",
    }
    buffer = WorkflowSpanBuffer(
        workflow_id="wf-file-1", run_id="run-1",
        workflow_type="FileWorkflow", task_queue="file-queue",
    )
    processor.get_buffer.return_value = buffer

    otel_setup._span_processor = processor
    hook_gov.configure(
        "http://localhost:9090", "test-key", processor,
        api_timeout=5.0, on_api_error=on_api_error,
    )
    otel_setup.setup_file_io_instrumentation()
    return processor


@contextmanager
def _mock_httpx_client(verdict="allow", reason=None, side_effect=None):
    """Mock persistent httpx client for governance API calls.

    Args:
        verdict: Governance verdict to return ("allow", "block", "halt").
        reason: Optional reason string for block/halt verdicts.
        side_effect: If set, post() raises this instead of returning a response.

    Yields:
        mock_instance: The mock client instance (access .post.call_args_list for assertions).
    """
    response = MagicMock()
    response.status_code = 200
    response_data = {"verdict": verdict}
    if reason:
        response_data["reason"] = reason
    response.json.return_value = response_data

    mock_instance = MagicMock()
    if side_effect:
        mock_instance.post.side_effect = side_effect
    else:
        mock_instance.post.return_value = response
    mock_instance.is_closed = False

    with patch("openbox.hook_governance._get_sync_client", return_value=mock_instance):
        yield mock_instance


class TestFileGovernanceStarted:
    """Tests for governance 'started' stage on file open."""

    def test_open_sends_started_governance(self):
        """Opening a file should send 'started' governance evaluation."""
        tmp_path = _make_temp_file(b"test data")
        _setup_governance()

        try:
            with _mock_httpx_client() as mock:
                with open(tmp_path, "r") as f:
                    f.read()

                assert mock.post.call_count >= 1
                payload = mock.post.call_args_list[0].kwargs["json"]
                assert payload["spans"][0]["hook_type"] == "file_operation"
                assert payload["spans"][0]["file_operation"] == "open"
                assert payload["spans"][0]["stage"] == "started"
                assert payload["spans"][0]["file_path"] == tmp_path
                assert payload["spans"][0]["file_mode"] == "r"
        finally:
            os.unlink(tmp_path)

    def test_open_blocked_raises_governance_error(self):
        """Opening a file blocked by governance should raise GovernanceBlockedError."""
        tmp_path = _make_temp_file()
        _setup_governance()

        try:
            with _mock_httpx_client(verdict="block", reason="Forbidden path"):
                with pytest.raises(GovernanceBlockedError) as exc_info:
                    open(tmp_path, "r")

                assert exc_info.value.verdict == Verdict.BLOCK
                assert exc_info.value.reason == "Forbidden path"
                assert exc_info.value.url == tmp_path
        finally:
            os.unlink(tmp_path)

    def test_open_halt_raises_governance_error(self):
        """HALT verdict should also block file access."""
        tmp_path = _make_temp_file()
        _setup_governance()

        try:
            with _mock_httpx_client(verdict="halt", reason="Security violation"):
                with pytest.raises(GovernanceBlockedError) as exc_info:
                    open(tmp_path, "r")

                assert exc_info.value.verdict == Verdict.HALT
        finally:
            os.unlink(tmp_path)


class TestFileGovernanceCompleted:
    """Tests for governance 'completed' stage on file close."""

    def test_close_sends_completed_governance_with_read_summary(self):
        """Closing a file should send 'completed' governance with operations summary."""
        tmp_path = _make_temp_file(b"hello world")
        _setup_governance()

        try:
            with _mock_httpx_client() as mock:
                with open(tmp_path, "r") as f:
                    content = f.read()

                payload = mock.post.call_args_list[-1].kwargs["json"]
                assert payload["spans"][0]["hook_type"] == "file_operation"
                assert payload["spans"][0]["file_operation"] == "close"
                assert payload["spans"][0]["stage"] == "completed"
                assert payload["spans"][0]["file_path"] == tmp_path
                assert payload["spans"][0]["bytes_read"] == len(content)
                assert payload["spans"][0]["bytes_written"] == 0
                assert "read" in payload["spans"][0]["operations"]
        finally:
            os.unlink(tmp_path)

    def test_close_reports_write_operations(self):
        """Completed stage should report write bytes and operations list."""
        tmp_path = _make_temp_file()
        _setup_governance()

        try:
            with _mock_httpx_client() as mock:
                with open(tmp_path, "w") as f:
                    f.write("hello")
                    f.write(" world")

                payload = mock.post.call_args_list[-1].kwargs["json"]
                assert payload["spans"][0]["bytes_written"] == len("hello") + len(" world")
                assert payload["spans"][0]["bytes_read"] == 0
                assert payload["spans"][0]["operations"] == ["write", "write"]
        finally:
            os.unlink(tmp_path)

    def test_started_and_completed_both_sent(self):
        """Open started + read started + read completed + close completed = 4 calls."""
        tmp_path = _make_temp_file(b"data")
        _setup_governance()

        try:
            with _mock_httpx_client() as mock:
                with open(tmp_path, "r") as f:
                    f.read()

                # open(started), read(started), read(completed), close(completed)
                assert mock.post.call_count == 4
                stages = [c.kwargs["json"]["spans"][0]["stage"] for c in mock.post.call_args_list]
                ops = [c.kwargs["json"]["spans"][0]["file_operation"] for c in mock.post.call_args_list]
                assert stages == ["started", "started", "completed", "completed"]
                assert ops == ["open", "read", "read", "close"]
        finally:
            os.unlink(tmp_path)


class TestFileGovernanceSkipPaths:
    """Tests for system path skipping."""

    def test_system_paths_skip_governance(self):
        """Files matching skip patterns should not trigger governance."""
        _setup_governance()

        with patch("openbox.hook_governance._get_sync_client") as mock_get:
            try:
                open("__pycache__/test.pyc", "r")
            except (FileNotFoundError, OSError):
                pass  # File doesn't exist — we just check no governance call

            mock_get.assert_not_called()


class TestFileGovernanceDisabled:
    """Tests for governance behavior when disabled."""

    def test_no_governance_outside_activity_context(self):
        """File ops outside activity context should not trigger governance."""
        tmp_path = _make_temp_file(b"data")
        processor = MagicMock()
        processor.get_activity_context_by_trace.return_value = None  # No activity

        otel_setup._span_processor = processor
        hook_gov.configure(
            "http://localhost:9090", "test-key", processor,
            api_timeout=5.0, on_api_error="fail_open",
        )
        otel_setup.setup_file_io_instrumentation()

        try:
            with patch("openbox.hook_governance._get_sync_client") as mock_get:
                with open(tmp_path, "r") as f:
                    f.read()
                mock_get.assert_not_called()
        finally:
            os.unlink(tmp_path)

    def test_no_governance_when_not_configured(self):
        """File governance should not fire when hook_governance is not configured."""
        tmp_path = _make_temp_file(b"data")
        otel_setup._span_processor = MagicMock()
        hook_gov._api_url = ""  # Not configured
        otel_setup.setup_file_io_instrumentation()

        try:
            with patch("openbox.hook_governance._get_sync_client") as mock_get:
                with open(tmp_path, "r") as f:
                    f.read()
                mock_get.assert_not_called()
        finally:
            os.unlink(tmp_path)


class TestFileGovernanceFailPolicy:
    """Tests for fail_open vs fail_closed on governance API errors."""

    def test_fail_open_allows_file_on_api_error(self):
        """With fail_open, file should open even if governance API fails."""
        tmp_path = _make_temp_file(b"test data")
        _setup_governance(on_api_error="fail_open")

        try:
            with _mock_httpx_client(side_effect=ConnectionError("API unavailable")):
                with open(tmp_path, "r") as f:
                    content = f.read()
                assert content == "test data"
        finally:
            os.unlink(tmp_path)

    def test_fail_closed_blocks_file_on_api_error(self):
        """With fail_closed, file should be blocked if governance API fails."""
        tmp_path = _make_temp_file(b"data")
        _setup_governance(on_api_error="fail_closed")

        try:
            with _mock_httpx_client(side_effect=ConnectionError("API unavailable")):
                with pytest.raises(GovernanceBlockedError):
                    open(tmp_path, "r")
        finally:
            os.unlink(tmp_path)


class TestFileReadGovernance:
    """Tests for per-operation read governance (started + completed)."""

    def test_read_started_and_completed(self):
        """read() should send started before and completed after the operation."""
        tmp_path = _make_temp_file(b"hello")
        _setup_governance()

        try:
            with _mock_httpx_client() as mock:
                with open(tmp_path, "r") as f:
                    f.read()

                calls = mock.post.call_args_list
                # Find read-specific calls (skip open started, close completed)
                read_calls = [c for c in calls if c.kwargs["json"]["spans"][0]["file_operation"] == "read"]
                assert len(read_calls) == 2
                assert read_calls[0].kwargs["json"]["spans"][0]["stage"] == "started"
                assert read_calls[1].kwargs["json"]["spans"][0]["stage"] == "completed"
        finally:
            os.unlink(tmp_path)

    def test_read_blocked_at_started(self):
        """Blocking verdict on read started should prevent the read."""
        tmp_path = _make_temp_file(b"secret data")
        _setup_governance()

        try:
            # Allow open, block on first read governance call
            call_count = {"n": 0}
            def conditional_response(*args, **kwargs):
                call_count["n"] += 1
                response = MagicMock()
                response.status_code = 200
                # First call = open started (allow), second = read started (block)
                if call_count["n"] == 2:
                    response.json.return_value = {"verdict": "block", "reason": "Read blocked"}
                else:
                    response.json.return_value = {"verdict": "allow"}
                return response

            mock_instance = MagicMock()
            mock_instance.post.side_effect = conditional_response
            mock_instance.is_closed = False

            with patch("openbox.hook_governance._get_sync_client", return_value=mock_instance):
                with pytest.raises(GovernanceBlockedError) as exc_info:
                    with open(tmp_path, "r") as f:
                        f.read()

                assert exc_info.value.verdict == Verdict.BLOCK
                assert exc_info.value.reason == "Read blocked"
        finally:
            os.unlink(tmp_path)

    def test_read_completed_contains_data(self):
        """read() completed trigger should include data and bytes_read."""
        tmp_path = _make_temp_file(b"payload content")
        _setup_governance()

        try:
            with _mock_httpx_client() as mock:
                with open(tmp_path, "r") as f:
                    f.read()

                read_completed = [
                    c for c in mock.post.call_args_list
                    if c.kwargs["json"]["spans"][0]["file_operation"] == "read"
                    and c.kwargs["json"]["spans"][0]["stage"] == "completed"
                ]
                assert len(read_completed) == 1
                trigger = read_completed[0].kwargs["json"]["spans"][0]
                assert trigger["data"] == "payload content"
                assert trigger["bytes_read"] == len("payload content")
        finally:
            os.unlink(tmp_path)


class TestFileWriteGovernance:
    """Tests for per-operation write governance (started + completed)."""

    def test_write_started_and_completed(self):
        """write() should send started before and completed after."""
        tmp_path = _make_temp_file()
        _setup_governance()

        try:
            with _mock_httpx_client() as mock:
                with open(tmp_path, "w") as f:
                    f.write("test output")

                write_calls = [
                    c for c in mock.post.call_args_list
                    if c.kwargs["json"]["spans"][0]["file_operation"] == "write"
                ]
                assert len(write_calls) == 2
                assert write_calls[0].kwargs["json"]["spans"][0]["stage"] == "started"
                assert write_calls[1].kwargs["json"]["spans"][0]["stage"] == "completed"
        finally:
            os.unlink(tmp_path)

    def test_write_blocked_at_started(self):
        """Blocking verdict on write started should prevent the write."""
        tmp_path = _make_temp_file()
        _setup_governance()

        try:
            call_count = {"n": 0}
            def conditional_response(*args, **kwargs):
                call_count["n"] += 1
                response = MagicMock()
                response.status_code = 200
                # First = open started (allow), second = write started (block)
                if call_count["n"] == 2:
                    response.json.return_value = {"verdict": "block", "reason": "Write denied"}
                else:
                    response.json.return_value = {"verdict": "allow"}
                return response

            mock_instance = MagicMock()
            mock_instance.post.side_effect = conditional_response
            mock_instance.is_closed = False

            with patch("openbox.hook_governance._get_sync_client", return_value=mock_instance):
                with pytest.raises(GovernanceBlockedError) as exc_info:
                    with open(tmp_path, "w") as f:
                        f.write("forbidden")

                assert exc_info.value.reason == "Write denied"
        finally:
            os.unlink(tmp_path)

    def test_write_completed_contains_data(self):
        """write() completed trigger should include data and bytes_written."""
        tmp_path = _make_temp_file()
        _setup_governance()

        try:
            with _mock_httpx_client() as mock:
                with open(tmp_path, "w") as f:
                    f.write("output data")

                write_completed = [
                    c for c in mock.post.call_args_list
                    if c.kwargs["json"]["spans"][0]["file_operation"] == "write"
                    and c.kwargs["json"]["spans"][0]["stage"] == "completed"
                ]
                assert len(write_completed) == 1
                trigger = write_completed[0].kwargs["json"]["spans"][0]
                assert trigger["data"] == "output data"
                assert trigger["bytes_written"] == len("output data")
        finally:
            os.unlink(tmp_path)


class TestFileReadlineGovernance:
    """Tests for readline governance (started + completed)."""

    def test_readline_started_and_completed(self):
        """readline() should send started + completed governance."""
        tmp_path = _make_temp_file(b"line1\nline2\n")
        _setup_governance()

        try:
            with _mock_httpx_client() as mock:
                with open(tmp_path, "r") as f:
                    f.readline()

                readline_calls = [
                    c for c in mock.post.call_args_list
                    if c.kwargs["json"]["spans"][0]["file_operation"] == "readline"
                ]
                assert len(readline_calls) == 2
                assert readline_calls[0].kwargs["json"]["spans"][0]["stage"] == "started"
                assert readline_calls[1].kwargs["json"]["spans"][0]["stage"] == "completed"
                assert readline_calls[1].kwargs["json"]["spans"][0]["data"] == "line1\n"
        finally:
            os.unlink(tmp_path)


class TestFileReadlinesGovernance:
    """Tests for readlines governance (started + completed)."""

    def test_readlines_started_and_completed(self):
        """readlines() should send started + completed with lines_count."""
        tmp_path = _make_temp_file(b"a\nb\nc\n")
        _setup_governance()

        try:
            with _mock_httpx_client() as mock:
                with open(tmp_path, "r") as f:
                    f.readlines()

                readlines_calls = [
                    c for c in mock.post.call_args_list
                    if c.kwargs["json"]["spans"][0]["file_operation"] == "readlines"
                ]
                assert len(readlines_calls) == 2
                assert readlines_calls[0].kwargs["json"]["spans"][0]["stage"] == "started"
                completed = readlines_calls[1].kwargs["json"]["spans"][0]
                assert completed["stage"] == "completed"
                assert completed["lines_count"] == 3
                assert completed["data"] == ["a\n", "b\n", "c\n"]
        finally:
            os.unlink(tmp_path)


class TestFileWritelinesGovernance:
    """Tests for writelines governance (started + completed)."""

    def test_writelines_started_and_completed(self):
        """writelines() should send started + completed with lines_count."""
        tmp_path = _make_temp_file()
        _setup_governance()

        try:
            with _mock_httpx_client() as mock:
                lines = ["x\n", "y\n"]
                with open(tmp_path, "w") as f:
                    f.writelines(lines)

                wl_calls = [
                    c for c in mock.post.call_args_list
                    if c.kwargs["json"]["spans"][0]["file_operation"] == "writelines"
                ]
                assert len(wl_calls) == 2
                assert wl_calls[0].kwargs["json"]["spans"][0]["stage"] == "started"
                completed = wl_calls[1].kwargs["json"]["spans"][0]
                assert completed["stage"] == "completed"
                assert completed["lines_count"] == 2
                assert completed["data"] == ["x\n", "y\n"]
                assert completed["bytes_written"] == 4
        finally:
            os.unlink(tmp_path)


class TestFileGovernanceCallCount:
    """Tests for total governance call counts per file lifecycle."""

    def test_open_read_close_produces_4_calls(self):
        """open(started) + read(started) + read(completed) + close(completed) = 4."""
        tmp_path = _make_temp_file(b"data")
        _setup_governance()

        try:
            with _mock_httpx_client() as mock:
                with open(tmp_path, "r") as f:
                    f.read()

                assert mock.post.call_count == 4
                ops = [c.kwargs["json"]["spans"][0]["file_operation"] for c in mock.post.call_args_list]
                assert ops == ["open", "read", "read", "close"]
        finally:
            os.unlink(tmp_path)

    def test_open_write_write_close_produces_6_calls(self):
        """open(started) + 2x write(started+completed) + close(completed) = 6."""
        tmp_path = _make_temp_file()
        _setup_governance()

        try:
            with _mock_httpx_client() as mock:
                with open(tmp_path, "w") as f:
                    f.write("a")
                    f.write("b")

                assert mock.post.call_count == 6
                ops = [c.kwargs["json"]["spans"][0]["file_operation"] for c in mock.post.call_args_list]
                assert ops == ["open", "write", "write", "write", "write", "close"]
        finally:
            os.unlink(tmp_path)


class TestFileGovernanceCompletedData:
    """Tests verifying data field in completed hook_trigger."""

    def test_read_completed_has_data_field(self):
        """read completed should have 'data' with the content read."""
        tmp_path = _make_temp_file(b"verification")
        _setup_governance()

        try:
            with _mock_httpx_client() as mock:
                with open(tmp_path, "r") as f:
                    f.read()

                read_completed = [
                    c for c in mock.post.call_args_list
                    if c.kwargs["json"]["spans"][0]["file_operation"] == "read"
                    and c.kwargs["json"]["spans"][0]["stage"] == "completed"
                ]
                assert read_completed[0].kwargs["json"]["spans"][0]["data"] == "verification"
        finally:
            os.unlink(tmp_path)

    def test_write_completed_has_data_field(self):
        """write completed should have 'data' with the content written."""
        tmp_path = _make_temp_file()
        _setup_governance()

        try:
            with _mock_httpx_client() as mock:
                with open(tmp_path, "w") as f:
                    f.write("output_data")

                write_completed = [
                    c for c in mock.post.call_args_list
                    if c.kwargs["json"]["spans"][0]["file_operation"] == "write"
                    and c.kwargs["json"]["spans"][0]["stage"] == "completed"
                ]
                assert write_completed[0].kwargs["json"]["spans"][0]["data"] == "output_data"
        finally:
            os.unlink(tmp_path)

    def test_started_has_no_data_field(self):
        """started triggers should NOT have a 'data' field."""
        tmp_path = _make_temp_file(b"check")
        _setup_governance()

        try:
            with _mock_httpx_client() as mock:
                with open(tmp_path, "r") as f:
                    f.read()

                started_calls = [
                    c for c in mock.post.call_args_list
                    if c.kwargs["json"]["spans"][0]["stage"] == "started"
                ]
                for call in started_calls:
                    assert "data" not in call.kwargs["json"]["spans"][0]
        finally:
            os.unlink(tmp_path)


class TestFileGovernanceSpanData:
    """Tests verifying span_data is included in governance payloads."""

    def test_open_payload_has_non_empty_spans(self):
        """Governance payload for open should include span data in spans array."""
        tmp_path = _make_temp_file(b"test data")
        _setup_governance()

        try:
            with _mock_httpx_client() as mock:
                with open(tmp_path, "r") as f:
                    f.read()

                # First call is open(started) — should have spans with file span data
                payload = mock.post.call_args_list[0].kwargs["json"]
                assert len(payload["spans"]) >= 1
                span_entry = payload["spans"][0]
                assert "span_id" in span_entry
                assert "trace_id" in span_entry
                assert span_entry["kind"] == "INTERNAL"
                assert span_entry["stage"] == "started"
                assert span_entry["file_path"] == tmp_path
        finally:
            os.unlink(tmp_path)

    def test_read_payload_has_spans(self):
        """Governance payload for read operations should include span data."""
        tmp_path = _make_temp_file(b"read me")
        _setup_governance()

        try:
            with _mock_httpx_client() as mock:
                with open(tmp_path, "r") as f:
                    f.read()

                # Find read started call
                read_started = [
                    c for c in mock.post.call_args_list
                    if c.kwargs["json"]["spans"][0]["file_operation"] == "read"
                    and c.kwargs["json"]["spans"][0]["stage"] == "started"
                ]
                assert len(read_started) == 1
                payload = read_started[0].kwargs["json"]
                assert payload["span_count"] >= 1
                # Each call sends only current span
                assert len(payload["spans"]) == 1
        finally:
            os.unlink(tmp_path)

    def test_span_data_has_correct_structure(self):
        """Span data entries should have required fields."""
        tmp_path = _make_temp_file(b"struct check")
        _setup_governance()

        try:
            with _mock_httpx_client() as mock:
                with open(tmp_path, "r") as f:
                    f.read()

                # Check any span entry for required fields
                payload = mock.post.call_args_list[0].kwargs["json"]
                span_entry = payload["spans"][0]
                required_fields = [
                    "span_id", "trace_id", "parent_span_id", "name",
                    "kind", "stage", "start_time", "attributes", "status",
                ]
                for field in required_fields:
                    assert field in span_entry, f"Missing field: {field}"
        finally:
            os.unlink(tmp_path)

    def test_write_payload_has_spans(self):
        """Governance payload for write operations should include span data."""
        tmp_path = _make_temp_file()
        _setup_governance()

        try:
            with _mock_httpx_client() as mock:
                with open(tmp_path, "w") as f:
                    f.write("data")

                write_calls = [
                    c for c in mock.post.call_args_list
                    if c.kwargs["json"]["spans"][0]["file_operation"] == "write"
                ]
                assert len(write_calls) >= 1
                payload = write_calls[0].kwargs["json"]
                assert len(payload["spans"]) >= 1
        finally:
            os.unlink(tmp_path)
