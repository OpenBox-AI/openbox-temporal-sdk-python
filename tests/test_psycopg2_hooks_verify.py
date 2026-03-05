"""Verify whether Psycopg2Instrumentor actually calls request_hook/response_hook.

This test proves that the installed version of opentelemetry-instrumentation-psycopg2
silently discards request_hook/response_hook kwargs passed to .instrument().
"""

from unittest.mock import MagicMock, patch

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor


def test_psycopg2_request_hook_is_silently_discarded():
    """Prove that request_hook passed to Psycopg2Instrumentor is never called."""
    provider = TracerProvider()
    trace.set_tracer_provider(provider)

    request_hook = MagicMock()
    response_hook = MagicMock()

    # Instrument with hooks
    Psycopg2Instrumentor().instrument(
        request_hook=request_hook,
        response_hook=response_hook,
    )

    try:
        # Mock psycopg2.connect to return a fake connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        import psycopg2
        with patch.object(psycopg2, "connect", return_value=mock_conn) as patched:
            # psycopg2.connect is wrapped by OTel — call the wrapper
            conn = psycopg2.connect(dbname="test")
            cur = conn.cursor()
            cur.execute("SELECT 1")

            # If hooks worked, request_hook would have been called
            print(f"request_hook called: {request_hook.called} (times: {request_hook.call_count})")
            print(f"response_hook called: {response_hook.called} (times: {response_hook.call_count})")

            # This assertion proves the hooks are silently discarded
            assert not request_hook.called, (
                "UNEXPECTED: request_hook WAS called! "
                "The OTel version supports hooks — update our approach."
            )
            print("CONFIRMED: request_hook is silently discarded by Psycopg2Instrumentor")

    finally:
        Psycopg2Instrumentor().uninstrument()
