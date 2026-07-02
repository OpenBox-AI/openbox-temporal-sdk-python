# tests/test_http_body_truncation.py
"""Tests for HTTP body truncation via max_body_size."""

import pytest
from unittest.mock import MagicMock, patch

# Import via otel_setup to avoid circular import
import openbox.otel_setup  # noqa: F401 — triggers full module init
from openbox.http_governance_hooks import _truncate_body, _build_http_span_data


class TestTruncateBody:
    def test_none_body_returns_none(self):
        assert _truncate_body(None, 100) is None

    def test_no_limit_returns_full_body(self):
        body = "x" * 10000
        assert _truncate_body(body, None) == body

    def test_body_under_limit_unchanged(self):
        assert _truncate_body("short", 100) == "short"

    def test_body_at_limit_unchanged(self):
        body = "x" * 100
        assert _truncate_body(body, 100) == body

    def test_body_over_limit_truncated(self):
        body = "x" * 200
        result = _truncate_body(body, 100)
        assert result.startswith("x" * 100)
        assert "[truncated" in result
        assert "200 total chars" in result

    def test_zero_limit_returns_full(self):
        """max_body_size=0 is falsy, treated as no limit."""
        assert _truncate_body("data", 0) == "data"


class TestBuildHttpSpanDataTruncation:
    def test_bodies_truncated_when_max_set(self):
        mock_span = MagicMock()
        mock_span.attributes = {}
        mock_span.name = "HTTP GET"
        large_body = "A" * 500

        with (
            patch("openbox.http_governance_hooks._hook_gov.get_max_body_size", return_value=100),
            patch("openbox.http_governance_hooks._hook_gov.extract_span_context", return_value=("s1", "t1", None)),
        ):
            result = _build_http_span_data(
                mock_span, "GET", "https://example.com", "completed",
                request_body=large_body, response_body=large_body,
            )

        assert len(result["request_body"]) < 500
        assert "[truncated" in result["request_body"]
        assert "[truncated" in result["response_body"]

    def test_bodies_not_truncated_when_no_limit(self):
        mock_span = MagicMock()
        mock_span.attributes = {}
        mock_span.name = "HTTP GET"
        large_body = "A" * 500

        with (
            patch("openbox.http_governance_hooks._hook_gov.get_max_body_size", return_value=None),
            patch("openbox.http_governance_hooks._hook_gov.extract_span_context", return_value=("s1", "t1", None)),
        ):
            result = _build_http_span_data(
                mock_span, "GET", "https://example.com", "completed",
                request_body=large_body, response_body=large_body,
            )

        assert result["request_body"] == large_body
        assert result["response_body"] == large_body


class TestMethodDecodingAndHeaderRedaction:
    """OTel's httpx instrumentation passes method as BYTES — str() mangled it
    into "b'POST'" on every httpx span. And raw request/response headers
    carried live credentials (authorization/cookie) into governance payloads."""

    def test_bytes_method_decodes_clean(self):
        from openbox.http_governance_hooks import _decode_method

        assert _decode_method(b"POST") == "POST"
        assert _decode_method("GET") == "GET"
        assert _decode_method(None) == "UNKNOWN"
        assert _decode_method(b"") == "UNKNOWN"

    def test_span_data_builder_decodes_bytes_method(self):
        from openbox.http_governance_hooks import _build_http_span_data

        span_data = _build_http_span_data(None, b"POST", "https://api.example.com", "started")
        assert span_data["http_method"] == "POST"

    def test_credential_headers_redacted_in_span_data(self):
        from openbox.http_governance_hooks import _build_http_span_data

        span_data = _build_http_span_data(
            None,
            "POST",
            "https://api.openai.com/v1/chat/completions",
            "started",
            request_headers={
                "Authorization": "Bearer sk-proj-SECRET",
                "Cookie": "__cf_bm=SECRET",
                "x-api-key": b"SECRET-BYTES",
                "content-type": "application/json",
            },
        )
        headers = span_data["request_headers"]
        assert headers["Authorization"] == "[REDACTED]"
        assert headers["Cookie"] == "[REDACTED]"
        assert headers["x-api-key"] == "[REDACTED]"
        assert headers["content-type"] == "application/json"
        assert "SECRET" not in str(span_data)

    def test_response_set_cookie_redacted(self):
        from openbox.http_governance_hooks import _build_http_span_data

        span_data = _build_http_span_data(
            None,
            "GET",
            "https://api.example.com",
            "completed",
            response_headers={"Set-Cookie": "session=SECRET", "server": "nginx"},
        )
        assert span_data["response_headers"]["Set-Cookie"] == "[REDACTED]"
        assert span_data["response_headers"]["server"] == "nginx"
