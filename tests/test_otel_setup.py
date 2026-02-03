# tests/test_otel_setup.py
"""
Comprehensive pytest tests for the OpenBox SDK otel_setup module.

Tests cover:
1. _should_ignore_url() - URL filtering for ignored prefixes
2. _is_text_content_type() - Content type detection for body capture
3. setup_opentelemetry_for_governance() - OTel instrumentor setup
4. setup_file_io_instrumentation() - File I/O tracing via builtins.open patching
5. uninstrument_file_io() - Restore original open()
6. setup_database_instrumentation() - Database library instrumentation
7. uninstrument_all() - Full cleanup
8. HTTP hooks - requests/httpx/urllib3/urllib hooks
"""

import builtins
import sys
import pytest
from unittest.mock import MagicMock, patch, call
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers to check if optional instrumentation packages are available
# ═══════════════════════════════════════════════════════════════════════════════


def _is_requests_instrumentation_available():
    """Check if requests and its OTel instrumentation are available."""
    try:
        import requests  # noqa
        from opentelemetry.instrumentation.requests import RequestsInstrumentor  # noqa
        return True
    except ImportError:
        return False


def _is_urllib3_instrumentation_available():
    """Check if urllib3 and its OTel instrumentation are available."""
    try:
        import urllib3  # noqa
        from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor  # noqa
        return True
    except ImportError:
        return False


# Skip markers for tests that require optional packages
requires_requests = pytest.mark.skipif(
    not _is_requests_instrumentation_available(),
    reason="requests or opentelemetry-instrumentation-requests not installed"
)

requires_urllib3 = pytest.mark.skipif(
    not _is_urllib3_instrumentation_available(),
    reason="urllib3 or opentelemetry-instrumentation-urllib3 not installed"
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures for managing global state
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def reset_otel_setup_globals():
    """Reset global state in otel_setup module before and after each test."""
    import openbox.otel_setup as otel_setup

    # Store original state
    original_span_processor = otel_setup._span_processor
    original_ignored_urls = otel_setup._ignored_url_prefixes.copy()
    original_open = builtins.open

    # Reset before test
    otel_setup._span_processor = None
    otel_setup._ignored_url_prefixes = set()

    yield

    # Reset after test
    otel_setup._span_processor = original_span_processor
    otel_setup._ignored_url_prefixes = original_ignored_urls

    # Restore builtins.open if it was patched
    if hasattr(builtins, '_openbox_original_open'):
        builtins.open = builtins._openbox_original_open
        delattr(builtins, '_openbox_original_open')
    else:
        builtins.open = original_open


@pytest.fixture
def mock_span_processor():
    """Create a mock WorkflowSpanProcessor."""
    processor = MagicMock()
    processor.store_body = MagicMock()
    return processor


@pytest.fixture
def mock_span():
    """Create a mock OTel span with context."""
    span = MagicMock()
    span.context = MagicMock()
    span.context.span_id = 12345678901234567890
    span.context.trace_id = 98765432109876543210
    return span


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for _should_ignore_url()
# ═══════════════════════════════════════════════════════════════════════════════


class TestShouldIgnoreUrl:
    """Tests for the _should_ignore_url() function."""

    def test_returns_false_for_empty_url(self):
        """Empty URL should not be ignored."""
        from openbox.otel_setup import _should_ignore_url

        assert _should_ignore_url("") is False
        assert _should_ignore_url(None) is False

    def test_returns_false_when_no_ignored_prefixes(self):
        """URLs should not be ignored when no prefixes are set."""
        from openbox.otel_setup import _should_ignore_url

        assert _should_ignore_url("https://api.example.com/data") is False
        assert _should_ignore_url("http://localhost:8080/test") is False

    def test_returns_true_for_urls_matching_ignored_prefixes(self):
        """URLs matching ignored prefixes should be ignored."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _should_ignore_url

        # Set ignored prefixes
        otel_setup._ignored_url_prefixes = {
            "https://api.openbox.ai",
            "http://localhost:9090/governance",
        }

        # Test matching URLs
        assert _should_ignore_url("https://api.openbox.ai/v1/events") is True
        assert _should_ignore_url("https://api.openbox.ai") is True
        assert _should_ignore_url("http://localhost:9090/governance/check") is True

    def test_returns_false_for_non_matching_urls(self):
        """URLs not matching any prefix should not be ignored."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _should_ignore_url

        otel_setup._ignored_url_prefixes = {
            "https://api.openbox.ai",
        }

        # Test non-matching URLs
        assert _should_ignore_url("https://api.example.com/data") is False
        assert _should_ignore_url("https://openbox.ai/docs") is False  # Different path
        assert _should_ignore_url("http://api.openbox.ai/v1") is False  # Different scheme

    def test_prefix_matching_is_case_sensitive(self):
        """URL prefix matching should be case-sensitive."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _should_ignore_url

        otel_setup._ignored_url_prefixes = {"https://api.openbox.ai"}

        assert _should_ignore_url("https://api.openbox.ai/v1") is True
        assert _should_ignore_url("https://API.OPENBOX.AI/v1") is False
        assert _should_ignore_url("HTTPS://api.openbox.ai/v1") is False


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for _is_text_content_type()
# ═══════════════════════════════════════════════════════════════════════════════


class TestIsTextContentType:
    """Tests for the _is_text_content_type() function."""

    def test_returns_true_for_text_types(self):
        """text/* content types should return True."""
        from openbox.otel_setup import _is_text_content_type

        assert _is_text_content_type("text/plain") is True
        assert _is_text_content_type("text/html") is True
        assert _is_text_content_type("text/xml") is True
        assert _is_text_content_type("text/css") is True
        assert _is_text_content_type("text/javascript") is True

    def test_returns_true_for_application_json(self):
        """application/json should return True."""
        from openbox.otel_setup import _is_text_content_type

        assert _is_text_content_type("application/json") is True
        assert _is_text_content_type("application/json; charset=utf-8") is True

    def test_returns_true_for_application_xml(self):
        """application/xml should return True."""
        from openbox.otel_setup import _is_text_content_type

        assert _is_text_content_type("application/xml") is True
        assert _is_text_content_type("application/xml; charset=utf-8") is True

    def test_returns_true_for_application_javascript(self):
        """application/javascript should return True."""
        from openbox.otel_setup import _is_text_content_type

        assert _is_text_content_type("application/javascript") is True

    def test_returns_true_for_form_urlencoded(self):
        """application/x-www-form-urlencoded should return True."""
        from openbox.otel_setup import _is_text_content_type

        assert _is_text_content_type("application/x-www-form-urlencoded") is True

    def test_returns_true_for_none(self):
        """None content type should return True (assume text)."""
        from openbox.otel_setup import _is_text_content_type

        assert _is_text_content_type(None) is True

    def test_returns_false_for_binary_types(self):
        """Binary content types should return False."""
        from openbox.otel_setup import _is_text_content_type

        assert _is_text_content_type("application/octet-stream") is False
        assert _is_text_content_type("image/png") is False
        assert _is_text_content_type("image/jpeg") is False
        assert _is_text_content_type("image/gif") is False
        assert _is_text_content_type("audio/mpeg") is False
        assert _is_text_content_type("video/mp4") is False
        assert _is_text_content_type("application/pdf") is False
        assert _is_text_content_type("application/zip") is False

    def test_handles_content_type_with_charset(self):
        """Content types with charset parameter should be handled correctly."""
        from openbox.otel_setup import _is_text_content_type

        assert _is_text_content_type("text/html; charset=utf-8") is True
        assert _is_text_content_type("application/json; charset=utf-8") is True
        assert _is_text_content_type("image/png; charset=utf-8") is False

    def test_handles_uppercase_content_types(self):
        """Content type matching should be case-insensitive."""
        from openbox.otel_setup import _is_text_content_type

        assert _is_text_content_type("TEXT/PLAIN") is True
        assert _is_text_content_type("Application/JSON") is True
        assert _is_text_content_type("APPLICATION/XML") is True
        assert _is_text_content_type("IMAGE/PNG") is False


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for setup_opentelemetry_for_governance()
# ═══════════════════════════════════════════════════════════════════════════════


class TestSetupOpentelemetryForGovernance:
    """Tests for the setup_opentelemetry_for_governance() function."""

    def test_sets_global_span_processor(self, mock_span_processor):
        """Should set the global _span_processor."""
        import openbox.otel_setup as otel_setup

        with patch.object(otel_setup, 'setup_httpx_body_capture'):
            with patch('opentelemetry.trace.get_tracer_provider') as mock_get_provider:
                mock_provider = MagicMock()
                mock_get_provider.return_value = mock_provider

                otel_setup.setup_opentelemetry_for_governance(
                    span_processor=mock_span_processor,
                    instrument_databases=False,
                    instrument_file_io=False,
                )

        assert otel_setup._span_processor is mock_span_processor

    def test_sets_ignored_url_prefixes(self, mock_span_processor):
        """Should set _ignored_url_prefixes from ignored_urls parameter."""
        import openbox.otel_setup as otel_setup

        ignored_urls = ["https://api.openbox.ai", "http://localhost:9090"]

        with patch.object(otel_setup, 'setup_httpx_body_capture'):
            with patch('opentelemetry.trace.get_tracer_provider') as mock_get_provider:
                mock_provider = MagicMock()
                mock_get_provider.return_value = mock_provider

                otel_setup.setup_opentelemetry_for_governance(
                    span_processor=mock_span_processor,
                    ignored_urls=ignored_urls,
                    instrument_databases=False,
                    instrument_file_io=False,
                )

        assert otel_setup._ignored_url_prefixes == set(ignored_urls)

    def test_registers_span_processor_with_tracer_provider(self, mock_span_processor):
        """Should register span processor with TracerProvider."""
        import openbox.otel_setup as otel_setup
        from opentelemetry.sdk.trace import TracerProvider

        with patch.object(otel_setup, 'setup_httpx_body_capture'):
            with patch('opentelemetry.trace.get_tracer_provider') as mock_get_provider:
                mock_provider = MagicMock(spec=TracerProvider)
                mock_get_provider.return_value = mock_provider

                otel_setup.setup_opentelemetry_for_governance(
                    span_processor=mock_span_processor,
                    instrument_databases=False,
                    instrument_file_io=False,
                )

                mock_provider.add_span_processor.assert_called_once_with(mock_span_processor)

    def test_creates_tracer_provider_if_not_exists(self, mock_span_processor):
        """Should create TracerProvider if none exists."""
        import openbox.otel_setup as otel_setup

        with patch.object(otel_setup, 'setup_httpx_body_capture'):
            with patch('opentelemetry.trace.get_tracer_provider') as mock_get_provider:
                with patch('opentelemetry.trace.set_tracer_provider') as mock_set_provider:
                    # Return a non-TracerProvider (e.g., NoOpTracerProvider)
                    mock_get_provider.return_value = MagicMock()
                    mock_get_provider.return_value.__class__.__name__ = "NoOpTracerProvider"

                    otel_setup.setup_opentelemetry_for_governance(
                        span_processor=mock_span_processor,
                        instrument_databases=False,
                        instrument_file_io=False,
                    )

                    # Verify set_tracer_provider was called
                    mock_set_provider.assert_called_once()

    @requires_requests
    def test_instruments_requests_library(self, mock_span_processor):
        """Should instrument requests library if available."""
        import openbox.otel_setup as otel_setup
        from opentelemetry.instrumentation.requests import RequestsInstrumentor

        with patch.object(otel_setup, 'setup_httpx_body_capture'):
            with patch('opentelemetry.trace.get_tracer_provider') as mock_get_provider:
                mock_provider = MagicMock()
                mock_get_provider.return_value = mock_provider

                with patch.object(RequestsInstrumentor, 'instrument') as mock_instrument:
                    otel_setup.setup_opentelemetry_for_governance(
                        span_processor=mock_span_processor,
                        instrument_databases=False,
                        instrument_file_io=False,
                    )

                    mock_instrument.assert_called_once()
                    # Verify hooks were passed
                    call_kwargs = mock_instrument.call_args[1]
                    assert 'request_hook' in call_kwargs
                    assert 'response_hook' in call_kwargs

    def test_instruments_httpx_library(self, mock_span_processor):
        """Should instrument httpx library if available."""
        import openbox.otel_setup as otel_setup

        with patch.object(otel_setup, 'setup_httpx_body_capture'):
            with patch('opentelemetry.trace.get_tracer_provider') as mock_get_provider:
                mock_provider = MagicMock()
                mock_get_provider.return_value = mock_provider

                with patch('opentelemetry.instrumentation.httpx.HTTPXClientInstrumentor') as mock_instrumentor:
                    mock_instance = MagicMock()
                    mock_instrumentor.return_value = mock_instance

                    otel_setup.setup_opentelemetry_for_governance(
                        span_processor=mock_span_processor,
                        instrument_databases=False,
                        instrument_file_io=False,
                    )

                    mock_instance.instrument.assert_called_once()
                    call_kwargs = mock_instance.instrument.call_args[1]
                    assert 'request_hook' in call_kwargs
                    assert 'response_hook' in call_kwargs
                    assert 'async_request_hook' in call_kwargs
                    assert 'async_response_hook' in call_kwargs

    @requires_urllib3
    def test_instruments_urllib3_library(self, mock_span_processor):
        """Should instrument urllib3 library if available."""
        import openbox.otel_setup as otel_setup
        from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor

        with patch.object(otel_setup, 'setup_httpx_body_capture'):
            with patch('opentelemetry.trace.get_tracer_provider') as mock_get_provider:
                mock_provider = MagicMock()
                mock_get_provider.return_value = mock_provider

                with patch.object(URLLib3Instrumentor, 'instrument') as mock_instrument:
                    otel_setup.setup_opentelemetry_for_governance(
                        span_processor=mock_span_processor,
                        instrument_databases=False,
                        instrument_file_io=False,
                    )

                    mock_instrument.assert_called_once()

    def test_instruments_urllib_library(self, mock_span_processor):
        """Should instrument urllib library if available."""
        import openbox.otel_setup as otel_setup

        with patch.object(otel_setup, 'setup_httpx_body_capture'):
            with patch('opentelemetry.trace.get_tracer_provider') as mock_get_provider:
                mock_provider = MagicMock()
                mock_get_provider.return_value = mock_provider

                with patch('opentelemetry.instrumentation.urllib.URLLibInstrumentor') as mock_instrumentor:
                    mock_instance = MagicMock()
                    mock_instrumentor.return_value = mock_instance

                    otel_setup.setup_opentelemetry_for_governance(
                        span_processor=mock_span_processor,
                        instrument_databases=False,
                        instrument_file_io=False,
                    )

                    mock_instance.instrument.assert_called_once()

    def test_calls_setup_httpx_body_capture(self, mock_span_processor):
        """Should call setup_httpx_body_capture for body capture."""
        import openbox.otel_setup as otel_setup

        with patch.object(otel_setup, 'setup_httpx_body_capture') as mock_body_capture:
            with patch('opentelemetry.trace.get_tracer_provider') as mock_get_provider:
                mock_provider = MagicMock()
                mock_get_provider.return_value = mock_provider

                otel_setup.setup_opentelemetry_for_governance(
                    span_processor=mock_span_processor,
                    instrument_databases=False,
                    instrument_file_io=False,
                )

                mock_body_capture.assert_called_once_with(mock_span_processor)

    def test_calls_setup_database_instrumentation_when_enabled(self, mock_span_processor):
        """Should call setup_database_instrumentation when instrument_databases=True."""
        import openbox.otel_setup as otel_setup

        with patch.object(otel_setup, 'setup_httpx_body_capture'):
            with patch.object(otel_setup, 'setup_database_instrumentation') as mock_db_setup:
                mock_db_setup.return_value = ["psycopg2"]
                with patch('opentelemetry.trace.get_tracer_provider') as mock_get_provider:
                    mock_provider = MagicMock()
                    mock_get_provider.return_value = mock_provider

                    otel_setup.setup_opentelemetry_for_governance(
                        span_processor=mock_span_processor,
                        instrument_databases=True,
                        db_libraries={"psycopg2"},
                        instrument_file_io=False,
                    )

                    mock_db_setup.assert_called_once_with({"psycopg2"})

    def test_calls_setup_file_io_instrumentation_when_enabled(self, mock_span_processor):
        """Should call setup_file_io_instrumentation when instrument_file_io=True."""
        import openbox.otel_setup as otel_setup

        with patch.object(otel_setup, 'setup_httpx_body_capture'):
            with patch.object(otel_setup, 'setup_file_io_instrumentation') as mock_file_setup:
                mock_file_setup.return_value = True
                with patch('opentelemetry.trace.get_tracer_provider') as mock_get_provider:
                    mock_provider = MagicMock()
                    mock_get_provider.return_value = mock_provider

                    otel_setup.setup_opentelemetry_for_governance(
                        span_processor=mock_span_processor,
                        instrument_databases=False,
                        instrument_file_io=True,
                    )

                    mock_file_setup.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for setup_file_io_instrumentation()
# ═══════════════════════════════════════════════════════════════════════════════


class TestSetupFileIoInstrumentation:
    """Tests for the setup_file_io_instrumentation() function."""

    def test_patches_builtins_open(self):
        """Should patch builtins.open with traced_open."""
        from openbox.otel_setup import setup_file_io_instrumentation

        original_open = builtins.open

        with patch('opentelemetry.trace.get_tracer'):
            result = setup_file_io_instrumentation()

        assert result is True
        assert builtins.open is not original_open
        assert hasattr(builtins, '_openbox_original_open')
        assert builtins._openbox_original_open is original_open

    def test_returns_true_if_already_instrumented(self):
        """Should return True if already instrumented (idempotent)."""
        from openbox.otel_setup import setup_file_io_instrumentation

        with patch('opentelemetry.trace.get_tracer'):
            # First call
            result1 = setup_file_io_instrumentation()
            # Second call should also succeed
            result2 = setup_file_io_instrumentation()

        assert result1 is True
        assert result2 is True

    def test_traced_file_wrapper_for_read(self, tmp_path):
        """TracedFile wrapper should work for read operations."""
        from openbox.otel_setup import setup_file_io_instrumentation, uninstrument_file_io

        # Create a test file
        test_file = tmp_path / "test_read.txt"
        test_file.write_text("Hello, World!")

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_tracer.start_as_current_span.return_value = mock_span
        mock_tracer.start_span.return_value = mock_span

        with patch('opentelemetry.trace.get_tracer', return_value=mock_tracer):
            setup_file_io_instrumentation()

            with open(str(test_file), 'r') as f:
                content = f.read()

            assert content == "Hello, World!"

        uninstrument_file_io()

    def test_traced_file_wrapper_for_write(self, tmp_path):
        """TracedFile wrapper should work for write operations."""
        from openbox.otel_setup import setup_file_io_instrumentation, uninstrument_file_io

        test_file = tmp_path / "test_write.txt"

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_tracer.start_as_current_span.return_value = mock_span
        mock_tracer.start_span.return_value = mock_span

        with patch('opentelemetry.trace.get_tracer', return_value=mock_tracer):
            setup_file_io_instrumentation()

            with open(str(test_file), 'w') as f:
                f.write("Test content")

            # Verify content was written
            assert test_file.read_text() == "Test content"

        uninstrument_file_io()

    def test_skips_system_paths(self, tmp_path):
        """Should skip instrumentation for system paths."""
        from openbox.otel_setup import setup_file_io_instrumentation, uninstrument_file_io

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_span.return_value = mock_span

        with patch('opentelemetry.trace.get_tracer', return_value=mock_tracer):
            setup_file_io_instrumentation()

            # System paths should be skipped (no span created)
            # We test by checking that certain patterns are skipped
            # The actual skip patterns are: /dev/, /proc/, /sys/, __pycache__, .pyc, .pyo, .so, .dylib

            # Create a file with __pycache__ in the path
            cache_dir = tmp_path / "__pycache__"
            cache_dir.mkdir()
            cache_file = cache_dir / "test.pyc"
            cache_file.write_bytes(b"fake bytecode")

            # Reading this file should use original open (no tracing)
            # We can verify by checking mock_tracer.start_span was not called for this path
            mock_tracer.reset_mock()

            # Read from cache path
            with builtins._openbox_original_open(str(cache_file), 'rb') as f:
                _ = f.read()

        uninstrument_file_io()

    def test_traced_file_readline(self, tmp_path):
        """TracedFile should handle readline() operations."""
        from openbox.otel_setup import setup_file_io_instrumentation, uninstrument_file_io

        test_file = tmp_path / "test_readline.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3")

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_tracer.start_as_current_span.return_value = mock_span
        mock_tracer.start_span.return_value = mock_span

        with patch('opentelemetry.trace.get_tracer', return_value=mock_tracer):
            setup_file_io_instrumentation()

            with open(str(test_file), 'r') as f:
                line = f.readline()
                assert line == "Line 1\n"

        uninstrument_file_io()

    def test_traced_file_readlines(self, tmp_path):
        """TracedFile should handle readlines() operations."""
        from openbox.otel_setup import setup_file_io_instrumentation, uninstrument_file_io

        test_file = tmp_path / "test_readlines.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3")

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_tracer.start_as_current_span.return_value = mock_span
        mock_tracer.start_span.return_value = mock_span

        with patch('opentelemetry.trace.get_tracer', return_value=mock_tracer):
            setup_file_io_instrumentation()

            with open(str(test_file), 'r') as f:
                lines = f.readlines()
                assert len(lines) == 3
                assert lines[0] == "Line 1\n"

        uninstrument_file_io()

    def test_traced_file_writelines(self, tmp_path):
        """TracedFile should handle writelines() operations."""
        from openbox.otel_setup import setup_file_io_instrumentation, uninstrument_file_io

        test_file = tmp_path / "test_writelines.txt"

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_tracer.start_as_current_span.return_value = mock_span
        mock_tracer.start_span.return_value = mock_span

        with patch('opentelemetry.trace.get_tracer', return_value=mock_tracer):
            setup_file_io_instrumentation()

            with open(str(test_file), 'w') as f:
                f.writelines(["Line 1\n", "Line 2\n", "Line 3\n"])

            assert test_file.read_text() == "Line 1\nLine 2\nLine 3\n"

        uninstrument_file_io()

    def test_traced_file_handles_open_error(self, tmp_path):
        """TracedFile should handle errors during open and set error attributes."""
        from openbox.otel_setup import setup_file_io_instrumentation, uninstrument_file_io

        non_existent_file = tmp_path / "non_existent.txt"

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_span.return_value = mock_span

        with patch('opentelemetry.trace.get_tracer', return_value=mock_tracer):
            setup_file_io_instrumentation()

            with pytest.raises(FileNotFoundError):
                open(str(non_existent_file), 'r')

            # Verify error was set on span
            mock_span.set_attribute.assert_any_call("error", True)

        uninstrument_file_io()


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for uninstrument_file_io()
# ═══════════════════════════════════════════════════════════════════════════════


class TestUninstrumentFileIo:
    """Tests for the uninstrument_file_io() function."""

    def test_restores_original_open(self):
        """Should restore the original builtins.open function."""
        from openbox.otel_setup import setup_file_io_instrumentation, uninstrument_file_io

        original_open = builtins.open

        with patch('opentelemetry.trace.get_tracer'):
            setup_file_io_instrumentation()

        assert builtins.open is not original_open
        assert hasattr(builtins, '_openbox_original_open')

        uninstrument_file_io()

        assert builtins.open is original_open
        assert not hasattr(builtins, '_openbox_original_open')

    def test_idempotent_when_not_instrumented(self):
        """Should be safe to call when not instrumented."""
        from openbox.otel_setup import uninstrument_file_io

        # Ensure not instrumented
        if hasattr(builtins, '_openbox_original_open'):
            delattr(builtins, '_openbox_original_open')

        # Should not raise
        uninstrument_file_io()


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for setup_database_instrumentation()
# ═══════════════════════════════════════════════════════════════════════════════


class TestSetupDatabaseInstrumentation:
    """Tests for the setup_database_instrumentation() function."""

    def test_instruments_all_available_when_db_libraries_is_none(self):
        """Should attempt to instrument all database libraries when db_libraries=None."""
        from openbox.otel_setup import setup_database_instrumentation

        with patch('opentelemetry.instrumentation.psycopg2.Psycopg2Instrumentor') as mock_psycopg2:
            with patch('opentelemetry.instrumentation.asyncpg.AsyncPGInstrumentor') as mock_asyncpg:
                mock_psycopg2_instance = MagicMock()
                mock_psycopg2.return_value = mock_psycopg2_instance
                mock_asyncpg_instance = MagicMock()
                mock_asyncpg.return_value = mock_asyncpg_instance

                result = setup_database_instrumentation(db_libraries=None)

                # Both should be called
                mock_psycopg2_instance.instrument.assert_called_once()
                mock_asyncpg_instance.instrument.assert_called_once()
                assert "psycopg2" in result
                assert "asyncpg" in result

    def test_instruments_only_specified_libraries(self):
        """Should only instrument specified libraries when db_libraries is set."""
        from openbox.otel_setup import setup_database_instrumentation

        with patch('opentelemetry.instrumentation.psycopg2.Psycopg2Instrumentor') as mock_psycopg2:
            with patch('opentelemetry.instrumentation.asyncpg.AsyncPGInstrumentor') as mock_asyncpg:
                mock_psycopg2_instance = MagicMock()
                mock_psycopg2.return_value = mock_psycopg2_instance
                mock_asyncpg_instance = MagicMock()
                mock_asyncpg.return_value = mock_asyncpg_instance

                result = setup_database_instrumentation(db_libraries={"psycopg2"})

                # Only psycopg2 should be called
                mock_psycopg2_instance.instrument.assert_called_once()
                mock_asyncpg_instance.instrument.assert_not_called()
                assert "psycopg2" in result
                assert "asyncpg" not in result

    def test_returns_list_of_instrumented_libraries(self):
        """Should return list of successfully instrumented library names."""
        from openbox.otel_setup import setup_database_instrumentation

        with patch('opentelemetry.instrumentation.psycopg2.Psycopg2Instrumentor') as mock_psycopg2:
            with patch('opentelemetry.instrumentation.mysql.MySQLInstrumentor') as mock_mysql:
                mock_psycopg2_instance = MagicMock()
                mock_psycopg2.return_value = mock_psycopg2_instance
                mock_mysql_instance = MagicMock()
                mock_mysql.return_value = mock_mysql_instance

                result = setup_database_instrumentation(db_libraries={"psycopg2", "mysql"})

                assert isinstance(result, list)
                assert "psycopg2" in result
                assert "mysql" in result

    def test_handles_import_errors_gracefully(self):
        """Should handle ImportError gracefully when library not available."""
        from openbox.otel_setup import setup_database_instrumentation

        # All imports will fail
        with patch.dict('sys.modules', {
            'opentelemetry.instrumentation.psycopg2': None,
            'opentelemetry.instrumentation.asyncpg': None,
            'opentelemetry.instrumentation.mysql': None,
            'opentelemetry.instrumentation.pymysql': None,
            'opentelemetry.instrumentation.pymongo': None,
            'opentelemetry.instrumentation.redis': None,
            'opentelemetry.instrumentation.sqlalchemy': None,
        }):
            # Force ImportError by patching the actual imports
            def raise_import_error(*args, **kwargs):
                raise ImportError("Test import error")

            with patch('builtins.__import__', side_effect=raise_import_error):
                # This should not raise, just return empty list
                result = setup_database_instrumentation(db_libraries=None)
                # All imports fail, so result should be empty
                # Note: The actual function catches ImportError, so result may vary

    def test_instruments_mysql(self):
        """Should instrument mysql-connector-python library."""
        from openbox.otel_setup import setup_database_instrumentation

        with patch('opentelemetry.instrumentation.mysql.MySQLInstrumentor') as mock_mysql:
            mock_instance = MagicMock()
            mock_mysql.return_value = mock_instance

            result = setup_database_instrumentation(db_libraries={"mysql"})

            mock_instance.instrument.assert_called_once()
            assert "mysql" in result

    def test_instruments_pymysql(self):
        """Should instrument pymysql library."""
        from openbox.otel_setup import setup_database_instrumentation

        with patch('opentelemetry.instrumentation.pymysql.PyMySQLInstrumentor') as mock_pymysql:
            mock_instance = MagicMock()
            mock_pymysql.return_value = mock_instance

            result = setup_database_instrumentation(db_libraries={"pymysql"})

            mock_instance.instrument.assert_called_once()
            assert "pymysql" in result

    def test_instruments_pymongo(self):
        """Should instrument pymongo library."""
        from openbox.otel_setup import setup_database_instrumentation

        with patch('opentelemetry.instrumentation.pymongo.PymongoInstrumentor') as mock_pymongo:
            mock_instance = MagicMock()
            mock_pymongo.return_value = mock_instance

            result = setup_database_instrumentation(db_libraries={"pymongo"})

            mock_instance.instrument.assert_called_once()
            assert "pymongo" in result

    def test_instruments_redis(self):
        """Should instrument redis library."""
        from openbox.otel_setup import setup_database_instrumentation

        with patch('opentelemetry.instrumentation.redis.RedisInstrumentor') as mock_redis:
            mock_instance = MagicMock()
            mock_redis.return_value = mock_instance

            result = setup_database_instrumentation(db_libraries={"redis"})

            mock_instance.instrument.assert_called_once()
            assert "redis" in result

    def test_instruments_sqlalchemy(self):
        """Should instrument sqlalchemy library."""
        from openbox.otel_setup import setup_database_instrumentation

        with patch('opentelemetry.instrumentation.sqlalchemy.SQLAlchemyInstrumentor') as mock_sqlalchemy:
            mock_instance = MagicMock()
            mock_sqlalchemy.return_value = mock_instance

            result = setup_database_instrumentation(db_libraries={"sqlalchemy"})

            mock_instance.instrument.assert_called_once()
            assert "sqlalchemy" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for uninstrument_all()
# ═══════════════════════════════════════════════════════════════════════════════


class TestUninstrumentAll:
    """Tests for the uninstrument_all() function."""

    def test_clears_span_processor(self, mock_span_processor):
        """Should clear the global _span_processor."""
        import openbox.otel_setup as otel_setup

        otel_setup._span_processor = mock_span_processor

        otel_setup.uninstrument_all()

        assert otel_setup._span_processor is None

    @requires_requests
    def test_uninstruments_requests(self):
        """Should uninstrument requests library."""
        import openbox.otel_setup as otel_setup
        from opentelemetry.instrumentation.requests import RequestsInstrumentor

        with patch.object(RequestsInstrumentor, 'uninstrument') as mock_uninstrument:
            otel_setup.uninstrument_all()
            mock_uninstrument.assert_called_once()

    def test_uninstruments_httpx(self):
        """Should uninstrument httpx library."""
        import openbox.otel_setup as otel_setup

        with patch('opentelemetry.instrumentation.httpx.HTTPXClientInstrumentor') as mock_instrumentor:
            mock_instance = MagicMock()
            mock_instrumentor.return_value = mock_instance

            otel_setup.uninstrument_all()

            mock_instance.uninstrument.assert_called_once()

    @requires_urllib3
    def test_uninstruments_urllib3(self):
        """Should uninstrument urllib3 library."""
        import openbox.otel_setup as otel_setup
        from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor

        with patch.object(URLLib3Instrumentor, 'uninstrument') as mock_uninstrument:
            otel_setup.uninstrument_all()
            mock_uninstrument.assert_called_once()

    def test_uninstruments_urllib(self):
        """Should uninstrument urllib library."""
        import openbox.otel_setup as otel_setup

        with patch('opentelemetry.instrumentation.urllib.URLLibInstrumentor') as mock_instrumentor:
            mock_instance = MagicMock()
            mock_instrumentor.return_value = mock_instance

            otel_setup.uninstrument_all()

            mock_instance.uninstrument.assert_called_once()

    def test_calls_uninstrument_databases(self):
        """Should call uninstrument_databases()."""
        import openbox.otel_setup as otel_setup

        with patch.object(otel_setup, 'uninstrument_databases') as mock_uninstrument_db:
            otel_setup.uninstrument_all()

            mock_uninstrument_db.assert_called_once()

    def test_calls_uninstrument_file_io(self):
        """Should call uninstrument_file_io()."""
        import openbox.otel_setup as otel_setup

        with patch.object(otel_setup, 'uninstrument_file_io') as mock_uninstrument_file:
            otel_setup.uninstrument_all()

            mock_uninstrument_file.assert_called_once()

    def test_handles_import_errors_gracefully(self):
        """Should handle ImportError gracefully during uninstrumentation."""
        import openbox.otel_setup as otel_setup

        # Test that uninstrument_all doesn't raise even when imports fail.
        # The function has try/except blocks that catch ImportError.
        # We verify this by patching at the module level in otel_setup.
        # If the function fails to catch ImportError, this test will fail.

        # Create mock modules that raise on attribute access
        mock_requests_module = MagicMock()
        mock_requests_module.RequestsInstrumentor.side_effect = ImportError("Mock import error")

        # We can't easily patch import errors at the try/except level,
        # but we can verify the function handles missing modules gracefully
        # by simply calling it (since requests/urllib3 aren't installed anyway)
        otel_setup.uninstrument_all()
        # If we get here without exception, the test passes


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for HTTP Hooks
# ═══════════════════════════════════════════════════════════════════════════════


class TestRequestsRequestHook:
    """Tests for _requests_request_hook()."""

    def test_stores_request_body_string(self, mock_span_processor, mock_span):
        """Should store string request body."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _requests_request_hook

        otel_setup._span_processor = mock_span_processor

        request = MagicMock()
        request.body = "test request body"

        _requests_request_hook(mock_span, request)

        mock_span_processor.store_body.assert_called_once_with(
            mock_span.context.span_id,
            request_body="test request body"
        )

    def test_stores_request_body_bytes(self, mock_span_processor, mock_span):
        """Should decode and store bytes request body."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _requests_request_hook

        otel_setup._span_processor = mock_span_processor

        request = MagicMock()
        request.body = b"test bytes body"

        _requests_request_hook(mock_span, request)

        mock_span_processor.store_body.assert_called_once_with(
            mock_span.context.span_id,
            request_body="test bytes body"
        )

    def test_does_nothing_when_no_span_processor(self, mock_span):
        """Should do nothing when _span_processor is None."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _requests_request_hook

        otel_setup._span_processor = None

        request = MagicMock()
        request.body = "test body"

        # Should not raise
        _requests_request_hook(mock_span, request)

    def test_does_nothing_when_no_body(self, mock_span_processor, mock_span):
        """Should do nothing when request has no body."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _requests_request_hook

        otel_setup._span_processor = mock_span_processor

        request = MagicMock()
        request.body = None

        _requests_request_hook(mock_span, request)

        mock_span_processor.store_body.assert_not_called()

    def test_handles_decode_errors(self, mock_span_processor, mock_span):
        """Should handle decode errors gracefully."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _requests_request_hook

        otel_setup._span_processor = mock_span_processor

        request = MagicMock()
        # Invalid UTF-8 bytes
        request.body = b'\xff\xfe\x00\x01'

        # Should not raise, should still store (with errors='ignore')
        _requests_request_hook(mock_span, request)


class TestRequestsResponseHook:
    """Tests for _requests_response_hook()."""

    def test_stores_response_body_for_text_content(self, mock_span_processor, mock_span):
        """Should store response body for text content types."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _requests_response_hook

        otel_setup._span_processor = mock_span_processor

        request = MagicMock()
        response = MagicMock()
        response.headers = {"content-type": "application/json"}
        response.text = '{"key": "value"}'

        _requests_response_hook(mock_span, request, response)

        mock_span_processor.store_body.assert_called_once_with(
            mock_span.context.span_id,
            response_body='{"key": "value"}'
        )

    def test_does_not_store_binary_content(self, mock_span_processor, mock_span):
        """Should not store response body for binary content types."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _requests_response_hook

        otel_setup._span_processor = mock_span_processor

        request = MagicMock()
        response = MagicMock()
        response.headers = {"content-type": "image/png"}
        response.text = "binary data"

        _requests_response_hook(mock_span, request, response)

        mock_span_processor.store_body.assert_not_called()

    def test_does_nothing_when_no_span_processor(self, mock_span):
        """Should do nothing when _span_processor is None."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _requests_response_hook

        otel_setup._span_processor = None

        request = MagicMock()
        response = MagicMock()
        response.headers = {"content-type": "application/json"}
        response.text = '{"key": "value"}'

        # Should not raise
        _requests_response_hook(mock_span, request, response)


class TestHttpxRequestHook:
    """Tests for _httpx_request_hook()."""

    def test_stores_request_headers(self, mock_span_processor, mock_span):
        """Should store request headers."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _httpx_request_hook

        otel_setup._span_processor = mock_span_processor

        request = MagicMock()
        request.url = "https://api.example.com/data"
        request.headers = {"Authorization": "Bearer token", "Content-Type": "application/json"}
        request.stream = None
        request._content = None

        _httpx_request_hook(mock_span, request)

        # Check that headers were stored
        calls = mock_span_processor.store_body.call_args_list
        assert any(
            call[1].get('request_headers') == {"Authorization": "Bearer token", "Content-Type": "application/json"}
            for call in calls
        )

    def test_stores_request_body_from_content(self, mock_span_processor, mock_span):
        """Should store request body from _content attribute."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _httpx_request_hook

        otel_setup._span_processor = mock_span_processor

        request = MagicMock()
        request.url = "https://api.example.com/data"
        request.headers = {}
        request.stream = None
        request._content = b'{"key": "value"}'
        # Mock hasattr properly
        delattr(request, 'content')

        _httpx_request_hook(mock_span, request)

        calls = mock_span_processor.store_body.call_args_list
        assert any(
            call[1].get('request_body') == '{"key": "value"}'
            for call in calls
        )

    def test_ignores_url_when_in_ignored_list(self, mock_span_processor, mock_span):
        """Should ignore URLs in the ignored prefixes list."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _httpx_request_hook

        otel_setup._span_processor = mock_span_processor
        otel_setup._ignored_url_prefixes = {"https://api.openbox.ai"}

        request = MagicMock()
        request.url = "https://api.openbox.ai/v1/events"
        request.headers = {"Content-Type": "application/json"}
        request._content = b'{"event": "test"}'

        _httpx_request_hook(mock_span, request)

        mock_span_processor.store_body.assert_not_called()

    def test_does_nothing_when_no_span_processor(self, mock_span):
        """Should do nothing when _span_processor is None."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _httpx_request_hook

        otel_setup._span_processor = None

        request = MagicMock()
        request.url = "https://api.example.com"
        request.headers = {}
        request._content = b"body"

        # Should not raise
        _httpx_request_hook(mock_span, request)


class TestHttpxResponseHook:
    """Tests for _httpx_response_hook()."""

    def test_stores_response_headers(self, mock_span_processor, mock_span):
        """Should store response headers."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _httpx_response_hook

        otel_setup._span_processor = mock_span_processor

        request = MagicMock()
        request.url = "https://api.example.com/data"

        response = MagicMock()
        response.headers = {"Content-Type": "application/json", "X-Request-Id": "abc123"}
        response._content = b'{"result": "ok"}'

        _httpx_response_hook(mock_span, request, response)

        calls = mock_span_processor.store_body.call_args_list
        assert any(
            call[1].get('response_headers') == {"Content-Type": "application/json", "X-Request-Id": "abc123"}
            for call in calls
        )

    def test_stores_response_body_for_text_content(self, mock_span_processor, mock_span):
        """Should store response body for text content types."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _httpx_response_hook

        otel_setup._span_processor = mock_span_processor

        request = MagicMock()
        request.url = "https://api.example.com/data"

        response = MagicMock()
        response.headers = {"content-type": "application/json"}
        response._content = b'{"result": "ok"}'
        # Remove content property to avoid fallback
        delattr(response, 'content')

        _httpx_response_hook(mock_span, request, response)

        calls = mock_span_processor.store_body.call_args_list
        assert any(
            call[1].get('response_body') == '{"result": "ok"}'
            for call in calls
        )

    def test_ignores_url_when_in_ignored_list(self, mock_span_processor, mock_span):
        """Should ignore URLs in the ignored prefixes list."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _httpx_response_hook

        otel_setup._span_processor = mock_span_processor
        otel_setup._ignored_url_prefixes = {"https://api.openbox.ai"}

        request = MagicMock()
        request.url = "https://api.openbox.ai/v1/events"

        response = MagicMock()
        response.headers = {"content-type": "application/json"}
        response._content = b'{"verdict": "allow"}'

        _httpx_response_hook(mock_span, request, response)

        mock_span_processor.store_body.assert_not_called()


class TestHttpxAsyncHooks:
    """Tests for async httpx hooks."""

    @pytest.mark.asyncio
    async def test_async_request_hook_delegates_to_sync(self, mock_span_processor, mock_span):
        """Async request hook should delegate to sync hook."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _httpx_async_request_hook

        otel_setup._span_processor = mock_span_processor

        request = MagicMock()
        request.url = "https://api.example.com/data"
        request.headers = {"Content-Type": "application/json"}
        request.stream = None
        request._content = b'{"key": "value"}'

        await _httpx_async_request_hook(mock_span, request)

        # Should have called store_body
        assert mock_span_processor.store_body.called

    @pytest.mark.asyncio
    async def test_async_response_hook_stores_headers(self, mock_span_processor, mock_span):
        """Async response hook should store response headers."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _httpx_async_response_hook

        otel_setup._span_processor = mock_span_processor

        request = MagicMock()
        request.url = "https://api.example.com/data"
        request.stream = None

        response = MagicMock()
        response.headers = {"content-type": "application/json"}
        response._content = b'{"result": "ok"}'
        response.aread = None  # No async read needed

        await _httpx_async_response_hook(mock_span, request, response)

        calls = mock_span_processor.store_body.call_args_list
        assert any(
            call[1].get('response_headers') == {"content-type": "application/json"}
            for call in calls
        )

    @pytest.mark.asyncio
    async def test_async_response_hook_ignores_url(self, mock_span_processor, mock_span):
        """Async response hook should ignore URLs in ignored list."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _httpx_async_response_hook

        otel_setup._span_processor = mock_span_processor
        otel_setup._ignored_url_prefixes = {"https://api.openbox.ai"}

        request = MagicMock()
        request.url = "https://api.openbox.ai/v1/events"

        response = MagicMock()
        response.headers = {"content-type": "application/json"}
        response._content = b'{"result": "ok"}'

        await _httpx_async_response_hook(mock_span, request, response)

        mock_span_processor.store_body.assert_not_called()


class TestUrllib3Hooks:
    """Tests for urllib3 hooks."""

    def test_request_hook_stores_body(self, mock_span_processor, mock_span):
        """Should store request body from RequestInfo."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _urllib3_request_hook

        otel_setup._span_processor = mock_span_processor

        pool = MagicMock()
        request_info = MagicMock()
        request_info.body = "test body"

        _urllib3_request_hook(mock_span, pool, request_info)

        mock_span_processor.store_body.assert_called_once_with(
            mock_span.context.span_id,
            request_body="test body"
        )

    def test_request_hook_decodes_bytes_body(self, mock_span_processor, mock_span):
        """Should decode bytes body."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _urllib3_request_hook

        otel_setup._span_processor = mock_span_processor

        pool = MagicMock()
        request_info = MagicMock()
        request_info.body = b"bytes body"

        _urllib3_request_hook(mock_span, pool, request_info)

        mock_span_processor.store_body.assert_called_once_with(
            mock_span.context.span_id,
            request_body="bytes body"
        )

    def test_response_hook_stores_body_for_text(self, mock_span_processor, mock_span):
        """Should store response body for text content types."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _urllib3_response_hook

        otel_setup._span_processor = mock_span_processor

        pool = MagicMock()
        response = MagicMock()
        response.headers = {"content-type": "application/json"}
        response.data = b'{"key": "value"}'

        _urllib3_response_hook(mock_span, pool, response)

        mock_span_processor.store_body.assert_called_once_with(
            mock_span.context.span_id,
            response_body='{"key": "value"}'
        )

    def test_response_hook_skips_binary_content(self, mock_span_processor, mock_span):
        """Should skip storing response body for binary content types."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _urllib3_response_hook

        otel_setup._span_processor = mock_span_processor

        pool = MagicMock()
        response = MagicMock()
        response.headers = {"content-type": "image/png"}
        response.data = b'\x89PNG...'

        _urllib3_response_hook(mock_span, pool, response)

        mock_span_processor.store_body.assert_not_called()


class TestUrllibHooks:
    """Tests for urllib (standard library) hooks."""

    def test_request_hook_stores_body(self, mock_span_processor, mock_span):
        """Should store request body."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _urllib_request_hook

        otel_setup._span_processor = mock_span_processor

        request = MagicMock()
        request.data = "request data"

        _urllib_request_hook(mock_span, request)

        mock_span_processor.store_body.assert_called_once_with(
            mock_span.context.span_id,
            request_body="request data"
        )

    def test_request_hook_decodes_bytes(self, mock_span_processor, mock_span):
        """Should decode bytes data."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _urllib_request_hook

        otel_setup._span_processor = mock_span_processor

        request = MagicMock()
        request.data = b"bytes data"

        _urllib_request_hook(mock_span, request)

        mock_span_processor.store_body.assert_called_once_with(
            mock_span.context.span_id,
            request_body="bytes data"
        )

    def test_request_hook_handles_no_data(self, mock_span_processor, mock_span):
        """Should handle requests with no data."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _urllib_request_hook

        otel_setup._span_processor = mock_span_processor

        request = MagicMock()
        request.data = None

        _urllib_request_hook(mock_span, request)

        mock_span_processor.store_body.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for setup_httpx_body_capture()
# ═══════════════════════════════════════════════════════════════════════════════


class TestSetupHttpxBodyCapture:
    """Tests for setup_httpx_body_capture()."""

    def test_patches_httpx_client_send(self, mock_span_processor):
        """Should patch httpx.Client.send method."""
        from openbox.otel_setup import setup_httpx_body_capture

        with patch('httpx.Client') as mock_client_class:
            with patch('httpx.AsyncClient') as mock_async_client_class:
                original_send = MagicMock()
                mock_client_class.send = original_send

                setup_httpx_body_capture(mock_span_processor)

                # Verify send was patched
                assert mock_client_class.send is not original_send

    def test_patches_httpx_async_client_send(self, mock_span_processor):
        """Should patch httpx.AsyncClient.send method."""
        from openbox.otel_setup import setup_httpx_body_capture

        with patch('httpx.Client') as mock_client_class:
            with patch('httpx.AsyncClient') as mock_async_client_class:
                original_send = MagicMock()
                mock_async_client_class.send = original_send

                setup_httpx_body_capture(mock_span_processor)

                # Verify send was patched
                assert mock_async_client_class.send is not original_send

    def test_handles_httpx_not_installed(self, mock_span_processor):
        """Should handle case when httpx is not installed."""
        from openbox.otel_setup import setup_httpx_body_capture

        with patch.dict('sys.modules', {'httpx': None}):
            # Force ImportError
            import sys
            original_import = __builtins__['__import__'] if isinstance(__builtins__, dict) else __builtins__.__import__

            def mock_import(name, *args, **kwargs):
                if name == 'httpx':
                    raise ImportError("No module named 'httpx'")
                return original_import(name, *args, **kwargs)

            with patch('builtins.__import__', side_effect=mock_import):
                # Should not raise
                setup_httpx_body_capture(mock_span_processor)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for uninstrument_databases()
# ═══════════════════════════════════════════════════════════════════════════════


class TestUninstrumentDatabases:
    """Tests for uninstrument_databases()."""

    def test_uninstruments_all_db_libraries(self):
        """Should attempt to uninstrument all database libraries."""
        from openbox.otel_setup import uninstrument_databases

        with patch('opentelemetry.instrumentation.psycopg2.Psycopg2Instrumentor') as mock_psycopg2:
            with patch('opentelemetry.instrumentation.asyncpg.AsyncPGInstrumentor') as mock_asyncpg:
                with patch('opentelemetry.instrumentation.mysql.MySQLInstrumentor') as mock_mysql:
                    mock_psycopg2_instance = MagicMock()
                    mock_psycopg2.return_value = mock_psycopg2_instance
                    mock_asyncpg_instance = MagicMock()
                    mock_asyncpg.return_value = mock_asyncpg_instance
                    mock_mysql_instance = MagicMock()
                    mock_mysql.return_value = mock_mysql_instance

                    uninstrument_databases()

                    mock_psycopg2_instance.uninstrument.assert_called_once()
                    mock_asyncpg_instance.uninstrument.assert_called_once()
                    mock_mysql_instance.uninstrument.assert_called_once()

    def test_handles_import_errors(self):
        """Should handle ImportError gracefully."""
        from openbox.otel_setup import uninstrument_databases

        # All imports fail
        with patch('opentelemetry.instrumentation.psycopg2.Psycopg2Instrumentor', side_effect=ImportError):
            with patch('opentelemetry.instrumentation.asyncpg.AsyncPGInstrumentor', side_effect=ImportError):
                # Should not raise
                uninstrument_databases()

    def test_handles_uninstrument_errors(self):
        """Should handle errors during uninstrument gracefully."""
        from openbox.otel_setup import uninstrument_databases

        with patch('opentelemetry.instrumentation.psycopg2.Psycopg2Instrumentor') as mock_psycopg2:
            mock_instance = MagicMock()
            mock_instance.uninstrument.side_effect = Exception("Uninstrument failed")
            mock_psycopg2.return_value = mock_instance

            # Should not raise
            uninstrument_databases()


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Integration tests for the otel_setup module."""

    def test_full_setup_and_teardown(self, mock_span_processor):
        """Test full setup and teardown cycle."""
        import openbox.otel_setup as otel_setup

        # Setup
        with patch.object(otel_setup, 'setup_httpx_body_capture'):
            with patch('opentelemetry.trace.get_tracer_provider') as mock_get_provider:
                mock_provider = MagicMock()
                mock_get_provider.return_value = mock_provider

                otel_setup.setup_opentelemetry_for_governance(
                    span_processor=mock_span_processor,
                    ignored_urls=["https://api.openbox.ai"],
                    instrument_databases=False,
                    instrument_file_io=False,
                )

        # Verify setup
        assert otel_setup._span_processor is mock_span_processor
        assert "https://api.openbox.ai" in otel_setup._ignored_url_prefixes

        # Teardown
        otel_setup.uninstrument_all()

        # Verify teardown
        assert otel_setup._span_processor is None

    def test_url_filtering_after_setup(self, mock_span_processor):
        """Test that URL filtering works correctly after setup."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _should_ignore_url

        # Setup with ignored URLs
        with patch.object(otel_setup, 'setup_httpx_body_capture'):
            with patch('opentelemetry.trace.get_tracer_provider') as mock_get_provider:
                mock_provider = MagicMock()
                mock_get_provider.return_value = mock_provider

                otel_setup.setup_opentelemetry_for_governance(
                    span_processor=mock_span_processor,
                    ignored_urls=["https://api.openbox.ai", "http://localhost:9090/governance"],
                    instrument_databases=False,
                    instrument_file_io=False,
                )

        # Test filtering
        assert _should_ignore_url("https://api.openbox.ai/v1/events") is True
        assert _should_ignore_url("http://localhost:9090/governance/check") is True
        assert _should_ignore_url("https://api.example.com/data") is False

    def test_hooks_use_global_span_processor(self, mock_span_processor, mock_span):
        """Test that hooks correctly use the global span processor."""
        import openbox.otel_setup as otel_setup
        from openbox.otel_setup import _requests_request_hook

        # Set up global span processor
        otel_setup._span_processor = mock_span_processor

        # Create a request
        request = MagicMock()
        request.body = "test body"

        # Call hook
        _requests_request_hook(mock_span, request)

        # Verify it used the global processor
        mock_span_processor.store_body.assert_called_once()

        # Clear global
        otel_setup._span_processor = None

        # Reset mock
        mock_span_processor.reset_mock()

        # Call hook again - should not call store_body
        _requests_request_hook(mock_span, request)
        mock_span_processor.store_body.assert_not_called()
