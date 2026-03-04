# openbox/otel_setup.py
"""
Setup OpenTelemetry instrumentors with body capture hooks.

Bodies are stored in the span processor buffer, NOT in OTel span attributes.
This keeps sensitive data out of external tracing systems while still
capturing it for governance evaluation.

Supported HTTP libraries:
- requests
- httpx (sync + async)
- urllib3
- urllib (standard library - request body only)

Supported database libraries:
- psycopg2 (PostgreSQL)
- asyncpg (PostgreSQL async)
- mysql-connector-python
- pymysql
- pymongo (MongoDB)
- redis
- sqlalchemy (ORM)
"""

from typing import TYPE_CHECKING, Any, Optional, Set, List
import contextvars
import logging

if TYPE_CHECKING:
    from .span_processor import WorkflowSpanProcessor

logger = logging.getLogger(__name__)

# Global reference to span processor for hooks
_span_processor: Optional["WorkflowSpanProcessor"] = None

# URLs to ignore (e.g., OpenBox Core API - we don't want to capture governance events)
_ignored_url_prefixes: Set[str] = set()

# Hook-level governance is handled by hook_governance module
from . import hook_governance as _hook_gov

# ContextVar to pass HTTP child span from OTel request hooks to _patched_send.
# Request hooks receive the correct HTTP span; we store it here so _patched_send
# can use it after _original_send() returns (when the HTTP span has ended and
# trace.get_current_span() would return the parent activity span).
_httpx_http_span: contextvars.ContextVar = contextvars.ContextVar(
    '_httpx_http_span', default=None
)

# Text content types that are safe to capture as body
_TEXT_CONTENT_TYPES = (
    "text/",
    "application/json",
    "application/xml",
    "application/javascript",
    "application/x-www-form-urlencoded",
)


def _should_ignore_url(url: str) -> bool:
    """Check if URL should be ignored (e.g., OpenBox Core API)."""
    if not url:
        return False
    for prefix in _ignored_url_prefixes:
        if url.startswith(prefix):
            return True
    return False


def _is_text_content_type(content_type: Optional[str]) -> bool:
    """Check if content type indicates text content (safe to decode)."""
    if not content_type:
        return True  # Assume text if no content-type
    content_type = content_type.lower().split(";")[0].strip()
    return any(content_type.startswith(t) for t in _TEXT_CONTENT_TYPES)



def _build_http_span_data(
    span,
    http_method: str,
    http_url: str,
    stage: str,
    request_body: Optional[str] = None,
    request_headers: Optional[dict] = None,
    response_body: Optional[str] = None,
    response_headers: Optional[dict] = None,
    http_status_code: Optional[int] = None,
) -> dict:
    """Build span data dict for an HTTP request (used by governance hooks).

    Creates a span data structure matching _extract_span_data() output,
    enriched with HTTP body/header data for governance evaluation.
    """
    import time as _time

    parent_span_id = None
    if span.parent and hasattr(span.parent, 'span_id') and isinstance(getattr(span.parent, 'span_id', None), int):
        parent_span_id = format(span.parent.span_id, "016x")

    attrs = dict(span.attributes) if hasattr(span, 'attributes') and span.attributes else {
        "http.method": http_method,
        "http.url": http_url,
    }
    # Enrich attributes with body/header data for governance evaluation
    if request_headers is not None:
        attrs["http.request.headers"] = request_headers
    if request_body is not None:
        attrs["http.request.body"] = request_body
    if response_headers is not None:
        attrs["http.response.headers"] = response_headers
    if response_body is not None:
        attrs["http.response.body"] = response_body
    if http_status_code is not None:
        attrs["http.response.status_code"] = http_status_code

    span_data = {
        "span_id": format(span.context.span_id, "016x"),
        "trace_id": format(span.context.trace_id, "032x"),
        "parent_span_id": parent_span_id,
        "name": span.name if hasattr(span, 'name') and span.name else f"HTTP {http_method}",
        "kind": "CLIENT",
        "stage": stage,
        "start_time": _time.time_ns(),
        "end_time": None,
        "duration_ns": None,
        "attributes": attrs,
        "status": {"code": "UNSET", "description": None},
        "events": [],
        "request_body": request_body,
        "response_body": response_body,
        "request_headers": request_headers,
        "response_headers": response_headers,
    }
    if http_status_code is not None:
        span_data["http_status_code"] = http_status_code
    return span_data


def setup_opentelemetry_for_governance(
    span_processor: "WorkflowSpanProcessor",
    api_url: str,
    api_key: str,
    *,
    ignored_urls: Optional[list] = None,
    instrument_databases: bool = True,
    db_libraries: Optional[Set[str]] = None,
    instrument_file_io: bool = False,
    sqlalchemy_engine: Optional[Any] = None,
    api_timeout: float = 30.0,
    on_api_error: str = "fail_open",
) -> None:
    """
    Setup OpenTelemetry instrumentors with body capture hooks.

    This function instruments HTTP, database, and file I/O libraries to:
    1. Create OTel spans for HTTP requests, database queries, and file operations
    2. Capture request/response bodies (via hooks that store in span_processor)
    3. Register the span processor with the OTel tracer provider

    Args:
        span_processor: The WorkflowSpanProcessor to store bodies in
        ignored_urls: List of URL prefixes to ignore (e.g., OpenBox Core API)
        instrument_databases: Whether to instrument database libraries (default: True)
        db_libraries: Set of database libraries to instrument (None = all available).
                      Valid values: "psycopg2", "asyncpg", "mysql", "pymysql",
                      "pymongo", "redis", "sqlalchemy"
        instrument_file_io: Whether to instrument file I/O operations (default: False)
        sqlalchemy_engine: Optional SQLAlchemy Engine instance to instrument. Required
                          when the engine is created before instrumentation runs (e.g.,
                          at module import time). If not provided, only future engines
                          created via create_engine() will be instrumented.
    """
    global _span_processor, _ignored_url_prefixes
    _span_processor = span_processor

    # Set ignored URL prefixes (always include api_url to prevent recursion)
    _ignored_url_prefixes = set(ignored_urls) if ignored_urls else set()
    _ignored_url_prefixes.add(api_url.rstrip("/"))
    logger.info(f"Ignoring URLs with prefixes: {_ignored_url_prefixes}")

    # Configure hook-level governance module (always enabled)
    _hook_gov.configure(
        api_url, api_key, span_processor,
        api_timeout=api_timeout, on_api_error=on_api_error,
    )

    # Register span processor with OTel tracer provider
    # This ensures on_end() is called when spans complete
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        # Create a new TracerProvider if none exists
        provider = TracerProvider()
        trace.set_tracer_provider(provider)

    provider.add_span_processor(span_processor)
    logger.info("Registered WorkflowSpanProcessor with OTel TracerProvider")

    # Track what was instrumented
    instrumented = []

    # 1. requests library
    try:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor

        RequestsInstrumentor().instrument(
            request_hook=_requests_request_hook,
            response_hook=_requests_response_hook,
        )
        instrumented.append("requests")
        logger.info("Instrumented: requests")
    except ImportError:
        logger.debug("requests instrumentation not available")

    # 2. httpx library (sync + async) - hooks for metadata only
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument(
            request_hook=_httpx_request_hook,
            response_hook=_httpx_response_hook,
            async_request_hook=_httpx_async_request_hook,
            async_response_hook=_httpx_async_response_hook,
        )
        instrumented.append("httpx")
        logger.info("Instrumented: httpx")
    except ImportError:
        logger.debug("httpx instrumentation not available")

    # 3. urllib3 library
    try:
        from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor

        URLLib3Instrumentor().instrument(
            request_hook=_urllib3_request_hook,
            response_hook=_urllib3_response_hook,
        )
        instrumented.append("urllib3")
        logger.info("Instrumented: urllib3")
    except ImportError:
        logger.debug("urllib3 instrumentation not available")

    # 4. urllib (standard library) - request body only, response body cannot be captured
    try:
        from opentelemetry.instrumentation.urllib import URLLibInstrumentor

        URLLibInstrumentor().instrument(
            request_hook=_urllib_request_hook,
        )
        instrumented.append("urllib")
        logger.info("Instrumented: urllib")
    except ImportError:
        logger.debug("urllib instrumentation not available")

    # 5. httpx body capture (separate from OTel - patches Client.send)
    setup_httpx_body_capture(span_processor)

    logger.info(f"OpenTelemetry HTTP instrumentation complete. Instrumented: {instrumented}")

    # 6. Database instrumentation (optional)
    if sqlalchemy_engine is not None and not instrument_databases:
        logger.warning(
            "sqlalchemy_engine was provided but instrument_databases=False; "
            "engine will not be instrumented"
        )
    if instrument_databases:
        db_instrumented = setup_database_instrumentation(db_libraries, sqlalchemy_engine)
        if db_instrumented:
            instrumented.extend(db_instrumented)

    # 7. File I/O instrumentation (optional)
    if instrument_file_io:
        if setup_file_io_instrumentation():
            instrumented.append("file_io")

    logger.info(f"OpenTelemetry governance setup complete. Instrumented: {instrumented}")


def setup_file_io_instrumentation() -> bool:
    """
    Setup file I/O instrumentation by patching built-in open().

    File operations will be captured as spans with:
    - file.path: File path
    - file.mode: Open mode (r, w, a, etc.)
    - file.operation: read, write, etc.
    - file.bytes: Number of bytes read/written

    Returns:
        True if instrumentation was successful
    """
    import builtins
    from opentelemetry import trace

    # Check if already instrumented
    if hasattr(builtins, '_openbox_original_open'):
        logger.debug("File I/O already instrumented")
        return True

    _original_open = builtins.open
    builtins._openbox_original_open = _original_open  # Store for uninstrumentation
    _tracer = trace.get_tracer("openbox.file_io")

    # Paths to skip (noisy system files)
    _skip_patterns = ('/dev/', '/proc/', '/sys/', '__pycache__', '.pyc', '.pyo', '.so', '.dylib')

    class TracedFile:
        """Wrapper around file object to trace read/write operations.

        Also sends hook-level governance evaluations:
        - "started" on open (can block file access)
        - "completed" on close (reports summary of operations performed)
        """

        def __init__(self, file_obj, file_path: str, mode: str, parent_span):
            self._file = file_obj
            self._file_path = file_path
            self._mode = mode
            self._parent_span = parent_span
            self._bytes_read = 0
            self._bytes_written = 0
            self._operations: list = []  # Track operations for governance payload

        def _evaluate_governance(self, operation: str, stage: str, **extra):
            """Send governance evaluation for a file operation stage."""
            if not _hook_gov.is_configured():
                return
            from .types import GovernanceBlockedError
            try:
                trigger = {
                    "type": "file_operation",
                    "operation": operation,
                    "file_path": self._file_path,
                    "file_mode": self._mode,
                    "stage": stage,
                    **extra,
                }
                _hook_gov.evaluate_sync(
                    self._parent_span,
                    hook_trigger=trigger,
                    identifier=self._file_path,
                )
            except GovernanceBlockedError:
                raise
            except Exception:
                pass  # fail_open handled inside evaluate_sync

        def read(self, size=-1):
            with _tracer.start_as_current_span("file.read") as span:
                span.set_attribute("file.path", self._file_path)
                span.set_attribute("file.operation", "read")
                self._evaluate_governance("read", "started")

                data = self._file.read(size)
                bytes_count = len(data) if isinstance(data, (str, bytes)) else 0
                self._bytes_read += bytes_count
                self._operations.append("read")
                span.set_attribute("file.bytes", bytes_count)

                self._evaluate_governance("read", "completed", data=data, bytes_read=bytes_count)
                return data

        def readline(self):
            with _tracer.start_as_current_span("file.readline") as span:
                span.set_attribute("file.path", self._file_path)
                span.set_attribute("file.operation", "readline")
                self._evaluate_governance("readline", "started")

                data = self._file.readline()
                bytes_count = len(data) if isinstance(data, (str, bytes)) else 0
                self._bytes_read += bytes_count
                self._operations.append("readline")
                span.set_attribute("file.bytes", bytes_count)

                self._evaluate_governance("readline", "completed", data=data, bytes_read=bytes_count)
                return data

        def readlines(self):
            with _tracer.start_as_current_span("file.readlines") as span:
                span.set_attribute("file.path", self._file_path)
                span.set_attribute("file.operation", "readlines")
                self._evaluate_governance("readlines", "started")

                data = self._file.readlines()
                bytes_count = sum(len(line) for line in data) if data else 0
                self._bytes_read += bytes_count
                self._operations.append("readlines")
                span.set_attribute("file.bytes", bytes_count)
                span.set_attribute("file.lines", len(data) if data else 0)

                self._evaluate_governance(
                    "readlines", "completed",
                    data=data, bytes_read=bytes_count,
                    lines_count=len(data) if data else 0,
                )
                return data

        def write(self, data):
            with _tracer.start_as_current_span("file.write") as span:
                span.set_attribute("file.path", self._file_path)
                span.set_attribute("file.operation", "write")
                self._evaluate_governance("write", "started")

                bytes_count = len(data) if isinstance(data, (str, bytes)) else 0
                span.set_attribute("file.bytes", bytes_count)
                self._bytes_written += bytes_count
                self._operations.append("write")
                result = self._file.write(data)

                self._evaluate_governance("write", "completed", data=data, bytes_written=bytes_count)
                return result

        def writelines(self, lines):
            with _tracer.start_as_current_span("file.writelines") as span:
                span.set_attribute("file.path", self._file_path)
                span.set_attribute("file.operation", "writelines")
                self._evaluate_governance("writelines", "started")

                bytes_count = sum(len(line) for line in lines) if lines else 0
                span.set_attribute("file.bytes", bytes_count)
                span.set_attribute("file.lines", len(lines) if lines else 0)
                self._bytes_written += bytes_count
                self._operations.append("writelines")
                result = self._file.writelines(lines)

                self._evaluate_governance(
                    "writelines", "completed",
                    data=lines, bytes_written=bytes_count,
                    lines_count=len(lines) if lines else 0,
                )
                return result

        def close(self):
            # Governance "completed" — reports what happened during file lifecycle
            # Use try/finally to ensure file handle and span are always cleaned up
            from .types import GovernanceBlockedError
            gov_error = None
            try:
                self._evaluate_governance(
                    "close", "completed",
                    bytes_read=self._bytes_read,
                    bytes_written=self._bytes_written,
                    operations=self._operations,
                )
            except GovernanceBlockedError as e:
                gov_error = e
            finally:
                if self._parent_span:
                    self._parent_span.set_attribute("file.total_bytes_read", self._bytes_read)
                    self._parent_span.set_attribute("file.total_bytes_written", self._bytes_written)
                    self._parent_span.end()
                self._file.close()
            if gov_error:
                raise gov_error

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            # Don't mask the original exception if close() also raises
            try:
                self.close()
            except Exception:
                if exc_type is None:
                    raise
            return False

        def __iter__(self):
            return iter(self._file)

        def __next__(self):
            return next(self._file)

        def __getattr__(self, name):
            return getattr(self._file, name)

    def traced_open(file, mode='r', *args, **kwargs):
        file_str = str(file)

        # Skip system/noisy paths
        if any(p in file_str for p in _skip_patterns):
            return _original_open(file, mode, *args, **kwargs)

        span = _tracer.start_span("file.open")
        span.set_attribute("file.path", file_str)
        span.set_attribute("file.mode", mode)

        # Governance "started" — can block file access before it happens
        if _hook_gov.is_configured():
            from .types import GovernanceBlockedError
            try:
                _hook_gov.evaluate_sync(
                    span,
                    hook_trigger={
                        "type": "file_operation",
                        "operation": "open",
                        "file_path": file_str,
                        "file_mode": mode,
                        "stage": "started",
                    },
                    identifier=file_str,
                )
            except GovernanceBlockedError:
                span.set_attribute("error", True)
                span.set_attribute("governance.blocked", True)
                span.end()
                raise
            except Exception:
                pass  # Non-governance errors are swallowed (fail_open handled inside evaluate_sync)

        try:
            file_obj = _original_open(file, mode, *args, **kwargs)
            return TracedFile(file_obj, file_str, mode, span)
        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            span.end()
            raise

    builtins.open = traced_open
    logger.info("Instrumented: file I/O (builtins.open)")
    return True


def uninstrument_file_io() -> None:
    """Restore original open() function."""
    import builtins
    if hasattr(builtins, '_openbox_original_open'):
        builtins.open = builtins._openbox_original_open
        delattr(builtins, '_openbox_original_open')
        logger.info("Uninstrumented: file I/O")


def setup_database_instrumentation(
    db_libraries: Optional[Set[str]] = None,
    sqlalchemy_engine: Optional[Any] = None,
) -> List[str]:
    """
    Setup OpenTelemetry database instrumentors.

    Database spans will be captured by the WorkflowSpanProcessor (already registered
    with the TracerProvider) and included in governance events.

    Args:
        db_libraries: Set of library names to instrument. If None, instruments all
                      available libraries. Valid values:
                      - "psycopg2" (PostgreSQL sync)
                      - "asyncpg" (PostgreSQL async)
                      - "mysql" (mysql-connector-python)
                      - "pymysql"
                      - "pymongo" (MongoDB)
                      - "redis"
                      - "sqlalchemy" (ORM)
        sqlalchemy_engine: Optional SQLAlchemy Engine instance to instrument. When
                          provided, registers event listeners on this engine to capture
                          queries. Without this, only engines created after this call
                          (via patched create_engine) will be instrumented.

    Returns:
        List of successfully instrumented library names
    """
    instrumented = []

    # psycopg2 (PostgreSQL sync)
    if db_libraries is None or "psycopg2" in db_libraries:
        try:
            from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor

            Psycopg2Instrumentor().instrument()
            instrumented.append("psycopg2")
            logger.info("Instrumented: psycopg2")
        except ImportError:
            logger.debug("psycopg2 instrumentation not available")

    # asyncpg (PostgreSQL async)
    if db_libraries is None or "asyncpg" in db_libraries:
        try:
            from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor

            AsyncPGInstrumentor().instrument()
            instrumented.append("asyncpg")
            logger.info("Instrumented: asyncpg")
        except ImportError:
            logger.debug("asyncpg instrumentation not available")

    # mysql-connector-python
    if db_libraries is None or "mysql" in db_libraries:
        try:
            from opentelemetry.instrumentation.mysql import MySQLInstrumentor

            MySQLInstrumentor().instrument()
            instrumented.append("mysql")
            logger.info("Instrumented: mysql")
        except ImportError:
            logger.debug("mysql instrumentation not available")

    # pymysql
    if db_libraries is None or "pymysql" in db_libraries:
        try:
            from opentelemetry.instrumentation.pymysql import PyMySQLInstrumentor

            PyMySQLInstrumentor().instrument()
            instrumented.append("pymysql")
            logger.info("Instrumented: pymysql")
        except ImportError:
            logger.debug("pymysql instrumentation not available")

    # pymongo (MongoDB)
    if db_libraries is None or "pymongo" in db_libraries:
        try:
            from opentelemetry.instrumentation.pymongo import PymongoInstrumentor

            PymongoInstrumentor().instrument()
            instrumented.append("pymongo")
            logger.info("Instrumented: pymongo")
        except ImportError:
            logger.debug("pymongo instrumentation not available")

    # redis
    if db_libraries is None or "redis" in db_libraries:
        try:
            from opentelemetry.instrumentation.redis import RedisInstrumentor

            RedisInstrumentor().instrument()
            instrumented.append("redis")
            logger.info("Instrumented: redis")
        except ImportError:
            logger.debug("redis instrumentation not available")

    # sqlalchemy (ORM)
    if sqlalchemy_engine is not None and db_libraries is not None and "sqlalchemy" not in db_libraries:
        logger.warning(
            "sqlalchemy_engine was provided but 'sqlalchemy' is not in db_libraries; "
            "engine will not be instrumented"
        )
    if db_libraries is None or "sqlalchemy" in db_libraries:
        try:
            from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

            if sqlalchemy_engine is not None:
                # Validate engine type before passing to instrumentor
                try:
                    from sqlalchemy.engine import Engine as _SAEngine
                except ImportError:
                    raise TypeError(
                        "sqlalchemy_engine was provided but sqlalchemy is not installed"
                    )
                if not isinstance(sqlalchemy_engine, _SAEngine):
                    raise TypeError(
                        f"sqlalchemy_engine must be a sqlalchemy.engine.Engine instance, "
                        f"got {type(sqlalchemy_engine).__name__}"
                    )
                # Instrument the existing engine directly (registers event listeners)
                SQLAlchemyInstrumentor().instrument(engine=sqlalchemy_engine)
                logger.info("Instrumented: sqlalchemy (existing engine)")
            else:
                # Patch create_engine() for future engines only
                SQLAlchemyInstrumentor().instrument()
                logger.info("Instrumented: sqlalchemy (future engines)")
            instrumented.append("sqlalchemy")
        except ImportError:
            logger.debug("sqlalchemy instrumentation not available")

    if instrumented:
        logger.info(f"Database instrumentation complete. Instrumented: {instrumented}")
    else:
        logger.debug("No database libraries instrumented (none available or installed)")

    return instrumented


def uninstrument_databases() -> None:
    """Uninstrument all database libraries."""
    try:
        from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor

        Psycopg2Instrumentor().uninstrument()
    except (ImportError, Exception):
        pass

    try:
        from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor

        AsyncPGInstrumentor().uninstrument()
    except (ImportError, Exception):
        pass

    try:
        from opentelemetry.instrumentation.mysql import MySQLInstrumentor

        MySQLInstrumentor().uninstrument()
    except (ImportError, Exception):
        pass

    try:
        from opentelemetry.instrumentation.pymysql import PyMySQLInstrumentor

        PyMySQLInstrumentor().uninstrument()
    except (ImportError, Exception):
        pass

    try:
        from opentelemetry.instrumentation.pymongo import PymongoInstrumentor

        PymongoInstrumentor().uninstrument()
    except (ImportError, Exception):
        pass

    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor

        RedisInstrumentor().uninstrument()
    except (ImportError, Exception):
        pass

    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        SQLAlchemyInstrumentor().uninstrument()
    except (ImportError, Exception):
        pass


def uninstrument_all() -> None:
    """Uninstrument all HTTP and database libraries."""
    global _span_processor
    _span_processor = None

    # Uninstrument HTTP libraries
    try:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor

        RequestsInstrumentor().uninstrument()
    except (ImportError, Exception):
        pass

    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().uninstrument()
    except (ImportError, Exception):
        pass

    try:
        from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor

        URLLib3Instrumentor().uninstrument()
    except (ImportError, Exception):
        pass

    try:
        from opentelemetry.instrumentation.urllib import URLLibInstrumentor

        URLLibInstrumentor().uninstrument()
    except (ImportError, Exception):
        pass

    # Uninstrument database libraries
    uninstrument_databases()

    # Uninstrument file I/O
    uninstrument_file_io()


# ═══════════════════════════════════════════════════════════════════════════════
# requests hooks
# ═══════════════════════════════════════════════════════════════════════════════


def _requests_request_hook(span, request) -> None:
    """
    Hook called before requests library sends a request.

    Args:
        span: OTel span
        request: requests.PreparedRequest
    """
    if _span_processor is None:
        return

    body = None
    try:
        if request.body:
            body = request.body
            if isinstance(body, bytes):
                body = body.decode("utf-8", errors="ignore")
    except Exception:
        pass

    if body:
        _span_processor.store_body(span.context.span_id, request_body=body)

    # Hook-level governance evaluation
    if _hook_gov.is_configured():
        url = str(request.url) if hasattr(request, 'url') else None
        if url and not _should_ignore_url(url):
            _span_processor.mark_governed(span.context.span_id)
            headers = dict(request.headers) if hasattr(request, 'headers') and request.headers else None
            method = request.method or "UNKNOWN"
            span_data = _build_http_span_data(span, method, url, "started", request_body=body, request_headers=headers)
            _hook_gov.evaluate_sync(
                span,
                hook_trigger={"type": "http_request", "method": method, "url": url, "stage": "started"},
                identifier=url,
                span_data=span_data,
            )


def _requests_response_hook(span, request, response) -> None:
    """
    Hook called after requests library receives a response.

    Args:
        span: OTel span
        request: requests.PreparedRequest
        response: requests.Response
    """
    if _span_processor is None:
        return

    resp_body = None
    resp_headers = None
    try:
        resp_headers = dict(response.headers) if hasattr(response, 'headers') and response.headers else None
        content_type = response.headers.get("content-type", "")
        if _is_text_content_type(content_type):
            resp_body = response.text
            _span_processor.store_body(span.context.span_id, response_body=resp_body)
    except Exception:
        pass

    # Hook-level governance evaluation (response stage)
    if _hook_gov.is_configured():
        url = str(request.url) if hasattr(request, 'url') else None
        if url and not _should_ignore_url(url):
            req_headers = dict(request.headers) if hasattr(request, 'headers') and request.headers else None
            req_body = None
            try:
                if request.body:
                    req_body = request.body
                    if isinstance(req_body, bytes):
                        req_body = req_body.decode("utf-8", errors="ignore")
            except Exception:
                pass
            method = request.method or "UNKNOWN"
            span_data = _build_http_span_data(
                span, method, url, "completed",
                request_body=req_body, request_headers=req_headers,
                response_body=resp_body, response_headers=resp_headers,
                http_status_code=getattr(response, 'status_code', None),
            )
            _hook_gov.evaluate_sync(
                span,
                hook_trigger={"type": "http_request", "method": method, "url": url, "stage": "completed"},
                identifier=url,
                span_data=span_data,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# httpx hooks
#
# These hooks are called by the OTel httpx instrumentation.
# We capture request/response bodies here for governance evaluation.
# ═══════════════════════════════════════════════════════════════════════════════


def _httpx_request_hook(span, request) -> None:
    """
    Hook called before httpx sends a request.

    Args:
        span: OTel span
        request: RequestInfo namedtuple with (method, url, headers, stream, extensions)
    """
    if _span_processor is None:
        return

    # Check if URL should be ignored
    url = str(request.url) if hasattr(request, 'url') else None
    if url and _should_ignore_url(url):
        return

    body = None
    request_headers = None
    try:
        # Capture request headers from RequestInfo namedtuple
        if hasattr(request, 'headers') and request.headers:
            request_headers = dict(request.headers)
            _span_processor.store_body(span.context.span_id, request_headers=request_headers)

        # Try to get request body - RequestInfo has a 'stream' attribute
        # httpx ByteStream stores body in _stream (not body or _body)
        if hasattr(request, 'stream'):
            stream = request.stream
            if hasattr(stream, '_stream') and isinstance(stream._stream, bytes):
                body = stream._stream
            elif hasattr(stream, 'body'):
                body = stream.body
            elif hasattr(stream, '_body'):
                body = stream._body
            elif isinstance(stream, bytes):
                body = stream

        # Fallback: Direct content attribute (for httpx.Request objects)
        if not body and hasattr(request, '_content') and request._content:
            body = request._content

        if not body and hasattr(request, 'content'):
            try:
                content = request.content
                if content:
                    body = content
            except Exception:
                pass

        if body:
            if isinstance(body, bytes):
                body = body.decode("utf-8", errors="ignore")
            elif not isinstance(body, str):
                body = str(body)
            _span_processor.store_body(span.context.span_id, request_body=body)

    except Exception:
        pass  # Best effort

    # Store HTTP child span so _patched_send can use it for governance
    _httpx_http_span.set(span)

    # Hook-level governance evaluation
    if _hook_gov.is_configured() and url:
        _span_processor.mark_governed(span.context.span_id)
        method = str(request.method) if hasattr(request, 'method') else "UNKNOWN"
        req_body = body if isinstance(body, str) else None
        span_data = _build_http_span_data(span, method, url, "started", request_body=req_body, request_headers=request_headers)
        _hook_gov.evaluate_sync(
            span,
            hook_trigger={"type": "http_request", "method": method, "url": url, "stage": "started"},
            identifier=url,
            span_data=span_data,
        )


def _httpx_response_hook(span, request, response) -> None:
    """
    Hook called after httpx receives a response.

    NOTE: At this point the response may not have been fully read yet.
    We try to read it here, but body capture may need to happen via
    the patched send method instead.

    Args:
        span: OTel span
        request: httpx.Request
        response: httpx.Response
    """
    if _span_processor is None:
        return

    # Check if URL should be ignored
    url = str(request.url) if hasattr(request, 'url') else None
    if url and _should_ignore_url(url):
        return

    resp_body = None
    resp_headers = None
    try:
        # Capture response headers first (always available even for streaming)
        if hasattr(response, 'headers') and response.headers:
            resp_headers = dict(response.headers)
            _span_processor.store_body(span.context.span_id, response_headers=resp_headers)

        content_type = response.headers.get("content-type", "")
        if _is_text_content_type(content_type):
            body = None

            # Check if response has already been read (has _content)
            if hasattr(response, '_content') and response._content:
                body = response._content
            # Try .content property
            elif hasattr(response, 'content'):
                try:
                    body = response.content
                except Exception:
                    pass

            if body:
                if isinstance(body, bytes):
                    body = body.decode("utf-8", errors="ignore")
                resp_body = body
                _span_processor.store_body(span.context.span_id, response_body=resp_body)
    except Exception:
        pass  # Best effort

    # NOTE: "completed" governance evaluation is handled in _patched_send
    # where request/response bodies are guaranteed to be available.


async def _httpx_async_request_hook(span, request) -> None:
    """Async version of request hook with async governance evaluation."""
    if _span_processor is None:
        return

    # Check if URL should be ignored
    url = str(request.url) if hasattr(request, 'url') else None
    if url and _should_ignore_url(url):
        return

    body = None
    request_headers = None
    try:
        # Capture request headers
        if hasattr(request, 'headers') and request.headers:
            request_headers = dict(request.headers)
            _span_processor.store_body(span.context.span_id, request_headers=request_headers)

        # Try to get request body
        if hasattr(request, 'stream'):
            stream = request.stream
            if hasattr(stream, 'body'):
                body = stream.body
            elif hasattr(stream, '_body'):
                body = stream._body
            elif isinstance(stream, bytes):
                body = stream

        if not body and hasattr(request, '_content') and request._content:
            body = request._content

        if not body and hasattr(request, 'content'):
            try:
                content = request.content
                if content:
                    body = content
            except Exception:
                pass

        if body:
            if isinstance(body, bytes):
                body = body.decode("utf-8", errors="ignore")
            elif not isinstance(body, str):
                body = str(body)
            _span_processor.store_body(span.context.span_id, request_body=body)

    except Exception:
        pass  # Best effort

    # Store HTTP child span so _patched_async_send can use it for governance
    _httpx_http_span.set(span)

    # Async hook-level governance evaluation
    if _hook_gov.is_configured() and url:
        _span_processor.mark_governed(span.context.span_id)
        method = str(request.method) if hasattr(request, 'method') else "UNKNOWN"
        req_body = body if isinstance(body, str) else None
        span_data = _build_http_span_data(span, method, url, "started", request_body=req_body, request_headers=request_headers)
        await _hook_gov.evaluate_async(
            span,
            hook_trigger={"type": "http_request", "method": method, "url": url, "stage": "started"},
            identifier=url,
            span_data=span_data,
        )


async def _httpx_async_response_hook(span, request, response) -> None:
    """Async version of response hook."""
    if _span_processor is None:
        return

    # Check if URL should be ignored
    url = str(request.url) if hasattr(request, 'url') else None
    if url and _should_ignore_url(url):
        return

    resp_body = None
    resp_headers = None
    try:
        # Capture response headers
        if hasattr(response, 'headers') and response.headers:
            resp_headers = dict(response.headers)
            _span_processor.store_body(span.context.span_id, response_headers=resp_headers)

        content_type = response.headers.get("content-type", "")
        if _is_text_content_type(content_type):
            body = None

            # Check if response has already been read
            if hasattr(response, '_content') and response._content:
                body = response._content
                if isinstance(body, bytes):
                    body = body.decode("utf-8", errors="ignore")
            # For async, try to read the response - THIS WILL CONSUME IT
            # but httpx caches it in _content after first read
            elif hasattr(response, 'aread'):
                try:
                    await response.aread()
                    if hasattr(response, '_content') and response._content:
                        body = response._content
                        if isinstance(body, bytes):
                            body = body.decode("utf-8", errors="ignore")
                except Exception:
                    pass

            if body:
                resp_body = body
                _span_processor.store_body(span.context.span_id, response_body=resp_body)

        # Also try to get request body from the stream
        request_body = None
        if hasattr(request, 'stream'):
            stream = request.stream
            if hasattr(stream, 'body'):
                request_body = stream.body
            elif hasattr(stream, '_body'):
                request_body = stream._body

        if request_body:
            if isinstance(request_body, bytes):
                request_body = request_body.decode("utf-8", errors="ignore")
            _span_processor.store_body(span.context.span_id, request_body=request_body)

    except Exception:
        pass  # Best effort

    # NOTE: "completed" governance evaluation is handled in _patched_async_send
    # where request/response bodies are guaranteed to be available.


# ═══════════════════════════════════════════════════════════════════════════════
# httpx body capture (patches Client.send)
# ═══════════════════════════════════════════════════════════════════════════════


def _capture_httpx_request_data(request) -> tuple:
    """Extract request body and headers from an httpx request.

    Returns:
        (request_body, request_headers) tuple. Either may be None.
    """
    request_body = None
    request_headers = None
    try:
        if hasattr(request, '_content') and request._content:
            request_body = request._content
            if isinstance(request_body, bytes):
                request_body = request_body.decode("utf-8", errors="ignore")
        elif hasattr(request, 'content') and request.content:
            request_body = request.content
            if isinstance(request_body, bytes):
                request_body = request_body.decode("utf-8", errors="ignore")
        if hasattr(request, 'headers') and request.headers:
            request_headers = dict(request.headers)
    except Exception as e:
        logger.debug(f"Failed to capture request body/headers: {e}")
    return request_body, request_headers


def _capture_httpx_response_data(response) -> tuple:
    """Extract response body and headers from an httpx response.

    Returns:
        (response_body, response_headers) tuple. Either may be None.
    """
    response_body = None
    response_headers = None
    content_type = response.headers.get("content-type", "")
    try:
        if hasattr(response, 'headers') and response.headers:
            response_headers = dict(response.headers)
        if _is_text_content_type(content_type):
            response_body = response.text
    except (UnicodeDecodeError, Exception) as e:
        logger.debug(f"Failed to capture response body: {e}")
    return response_body, response_headers


def _get_httpx_http_span():
    """Retrieve and reset the HTTP span stored by request hooks.

    Falls back to the current OTel span if no stored span is found.
    """
    http_span = _httpx_http_span.get(None)
    _httpx_http_span.set(None)
    if http_span is None:
        from opentelemetry import trace
        http_span = trace.get_current_span()
    return http_span


def _store_body_data(span_processor, http_span, request_body, request_headers,
                     response_body, response_headers) -> None:
    """Store captured HTTP body/header data against the span."""
    try:
        if http_span and hasattr(http_span, 'context') and http_span.context.span_id:
            span_processor.store_body(
                http_span.context.span_id,
                request_body=request_body or None,
                response_body=response_body or None,
                request_headers=request_headers or None,
                response_headers=response_headers or None,
            )
    except Exception as e:
        logger.debug(f"Failed to store body: {e}")


def _prepare_completed_governance(http_span, request, url, request_body, request_headers,
                                  response_body, response_headers, status_code):
    """Build 'completed' governance args. Returns tuple or None if not applicable."""
    if not (_hook_gov.is_configured() and url and http_span):
        return None
    method = str(request.method) if hasattr(request, 'method') else "UNKNOWN"
    span_data = _build_http_span_data(
        http_span, method, url, "completed",
        request_body=request_body, request_headers=request_headers,
        response_body=response_body, response_headers=response_headers,
        http_status_code=status_code,
    )
    hook_trigger = {"type": "http_request", "method": method, "url": url, "stage": "completed"}
    return http_span, hook_trigger, url, span_data


def setup_httpx_body_capture(span_processor: "WorkflowSpanProcessor") -> None:
    """
    Setup httpx body capture using Client.send patching.

    This is separate from OTel instrumentation because OTel hooks
    receive streams that cannot be safely consumed.
    """
    try:
        import httpx

        _original_send = httpx.Client.send
        _original_async_send = httpx.AsyncClient.send

        def _patched_send(self, request, *args, **kwargs):
            url = str(request.url) if hasattr(request, 'url') else None
            if url and _should_ignore_url(url):
                return _original_send(self, request, *args, **kwargs)

            request_body, request_headers = _capture_httpx_request_data(request)
            response = _original_send(self, request, *args, **kwargs)
            http_span = _get_httpx_http_span()
            response_body, response_headers = _capture_httpx_response_data(response)

            _store_body_data(span_processor, http_span, request_body, request_headers,
                             response_body, response_headers)
            gov_args = _prepare_completed_governance(
                http_span, request, url, request_body, request_headers,
                response_body, response_headers, getattr(response, 'status_code', None),
            )
            if gov_args:
                _hook_gov.evaluate_sync(gov_args[0], gov_args[1], gov_args[2], span_data=gov_args[3])
            return response

        async def _patched_async_send(self, request, *args, **kwargs):
            url = str(request.url) if hasattr(request, 'url') else None
            if url and _should_ignore_url(url):
                return await _original_async_send(self, request, *args, **kwargs)

            request_body, request_headers = _capture_httpx_request_data(request)
            response = await _original_async_send(self, request, *args, **kwargs)
            http_span = _get_httpx_http_span()
            response_body, response_headers = _capture_httpx_response_data(response)

            _store_body_data(span_processor, http_span, request_body, request_headers,
                             response_body, response_headers)
            gov_args = _prepare_completed_governance(
                http_span, request, url, request_body, request_headers,
                response_body, response_headers, getattr(response, 'status_code', None),
            )
            if gov_args:
                await _hook_gov.evaluate_async(gov_args[0], gov_args[1], gov_args[2], span_data=gov_args[3])
            return response

        httpx.Client.send = _patched_send
        httpx.AsyncClient.send = _patched_async_send
        logger.info("Patched httpx for body capture")

    except ImportError:
        logger.debug("httpx not available for body capture")


# ═══════════════════════════════════════════════════════════════════════════════
# urllib3 hooks
# ═══════════════════════════════════════════════════════════════════════════════


def _urllib3_request_hook(span, pool, request_info) -> None:
    """
    Hook called before urllib3 sends a request.

    Args:
        span: OTel span
        pool: urllib3.HTTPConnectionPool
        request_info: RequestInfo namedtuple
    """
    if _span_processor is None:
        return

    body = None
    try:
        if hasattr(request_info, "body") and request_info.body:
            body = request_info.body
            if isinstance(body, bytes):
                body = body.decode("utf-8", errors="ignore")
            _span_processor.store_body(span.context.span_id, request_body=body)
    except Exception:
        pass

    # Hook-level governance evaluation
    if _hook_gov.is_configured():
        # Reconstruct URL from pool and request_info
        scheme = getattr(pool, 'scheme', 'http')
        host = getattr(pool, 'host', 'unknown')
        port = getattr(pool, 'port', None)
        url_path = getattr(request_info, 'url', getattr(request_info, 'request_url', '/'))
        if port and port not in (80, 443):
            url = f"{scheme}://{host}:{port}{url_path}"
        else:
            url = f"{scheme}://{host}{url_path}"

        if not _should_ignore_url(url):
            _span_processor.mark_governed(span.context.span_id)
            method = getattr(request_info, 'method', 'UNKNOWN')
            headers = dict(request_info.headers) if hasattr(request_info, 'headers') and request_info.headers else None
            req_body = body if isinstance(body, str) else None
            span_data = _build_http_span_data(span, method, url, "started", request_body=req_body, request_headers=headers)
            _hook_gov.evaluate_sync(
                span,
                hook_trigger={"type": "http_request", "method": method, "url": url, "stage": "started"},
                identifier=url,
                span_data=span_data,
            )


def _urllib3_response_hook(span, pool, response) -> None:
    """
    Hook called after urllib3 receives a response.

    Args:
        span: OTel span
        pool: urllib3.HTTPConnectionPool
        response: urllib3.HTTPResponse
    """
    if _span_processor is None:
        return

    resp_body = None
    resp_headers = None
    try:
        resp_headers = dict(response.headers) if hasattr(response, 'headers') and response.headers else None
        content_type = response.headers.get("content-type", "")
        if _is_text_content_type(content_type):
            body = response.data
            if isinstance(body, bytes):
                body = body.decode("utf-8", errors="ignore")
            if body:
                resp_body = body
                _span_processor.store_body(span.context.span_id, response_body=resp_body)
    except Exception:
        pass

    # Hook-level governance evaluation (response stage)
    if _hook_gov.is_configured():
        # Reconstruct URL from pool
        scheme = getattr(pool, 'scheme', 'http')
        host = getattr(pool, 'host', 'unknown')
        port = getattr(pool, 'port', None)
        if port and port not in (80, 443):
            url = f"{scheme}://{host}:{port}/"
        else:
            url = f"{scheme}://{host}/"

        if not _should_ignore_url(url):
            status_code = getattr(response, 'status', None)
            span_data = _build_http_span_data(
                span, "UNKNOWN", url, "completed",
                response_body=resp_body, response_headers=resp_headers,
                http_status_code=status_code,
            )
            _hook_gov.evaluate_sync(
                span,
                hook_trigger={"type": "http_request", "method": "UNKNOWN", "url": url, "stage": "completed"},
                identifier=url,
                span_data=span_data,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# urllib hooks (standard library)
# NOTE: Response body capture is NOT supported - read() consumes the socket stream
# ═══════════════════════════════════════════════════════════════════════════════


def _urllib_request_hook(span, request) -> None:
    """Hook called before urllib sends a request."""
    if _span_processor is None:
        return

    try:
        if request.data:
            body = request.data
            if isinstance(body, bytes):
                body = body.decode("utf-8", errors="ignore")
            _span_processor.store_body(span.context.span_id, request_body=body)
    except Exception:
        pass