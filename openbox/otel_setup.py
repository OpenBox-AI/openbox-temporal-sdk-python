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

from typing import TYPE_CHECKING, Optional, Set, List
import logging

if TYPE_CHECKING:
    from .span_processor import WorkflowSpanProcessor

logger = logging.getLogger(__name__)

# Global reference to span processor for hooks
_span_processor: Optional["WorkflowSpanProcessor"] = None

# URLs to ignore (e.g., OpenBox Core API - we don't want to capture governance events)
_ignored_url_prefixes: Set[str] = set()

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


def setup_opentelemetry_for_governance(
    span_processor: "WorkflowSpanProcessor",
    ignored_urls: Optional[list] = None,
    instrument_databases: bool = True,
    db_libraries: Optional[Set[str]] = None,
    instrument_file_io: bool = False,
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
    """
    global _span_processor, _ignored_url_prefixes
    _span_processor = span_processor

    # Set ignored URL prefixes
    if ignored_urls:
        _ignored_url_prefixes = set(ignored_urls)
        logger.info(f"Ignoring URLs with prefixes: {_ignored_url_prefixes}")

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
    if instrument_databases:
        db_instrumented = setup_database_instrumentation(db_libraries)
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
        """Wrapper around file object to trace read/write operations."""

        def __init__(self, file_obj, file_path: str, mode: str, parent_span):
            self._file = file_obj
            self._file_path = file_path
            self._mode = mode
            self._parent_span = parent_span
            self._bytes_read = 0
            self._bytes_written = 0

        def read(self, size=-1):
            with _tracer.start_as_current_span("file.read") as span:
                span.set_attribute("file.path", self._file_path)
                span.set_attribute("file.operation", "read")
                data = self._file.read(size)
                bytes_count = len(data) if isinstance(data, (str, bytes)) else 0
                self._bytes_read += bytes_count
                span.set_attribute("file.bytes", bytes_count)
                return data

        def readline(self):
            with _tracer.start_as_current_span("file.readline") as span:
                span.set_attribute("file.path", self._file_path)
                span.set_attribute("file.operation", "readline")
                data = self._file.readline()
                bytes_count = len(data) if isinstance(data, (str, bytes)) else 0
                self._bytes_read += bytes_count
                span.set_attribute("file.bytes", bytes_count)
                return data

        def readlines(self):
            with _tracer.start_as_current_span("file.readlines") as span:
                span.set_attribute("file.path", self._file_path)
                span.set_attribute("file.operation", "readlines")
                data = self._file.readlines()
                bytes_count = sum(len(line) for line in data) if data else 0
                self._bytes_read += bytes_count
                span.set_attribute("file.bytes", bytes_count)
                span.set_attribute("file.lines", len(data) if data else 0)
                return data

        def write(self, data):
            with _tracer.start_as_current_span("file.write") as span:
                span.set_attribute("file.path", self._file_path)
                span.set_attribute("file.operation", "write")
                bytes_count = len(data) if isinstance(data, (str, bytes)) else 0
                span.set_attribute("file.bytes", bytes_count)
                self._bytes_written += bytes_count
                return self._file.write(data)

        def writelines(self, lines):
            with _tracer.start_as_current_span("file.writelines") as span:
                span.set_attribute("file.path", self._file_path)
                span.set_attribute("file.operation", "writelines")
                bytes_count = sum(len(line) for line in lines) if lines else 0
                span.set_attribute("file.bytes", bytes_count)
                span.set_attribute("file.lines", len(lines) if lines else 0)
                self._bytes_written += bytes_count
                return self._file.writelines(lines)

        def close(self):
            if self._parent_span:
                self._parent_span.set_attribute("file.total_bytes_read", self._bytes_read)
                self._parent_span.set_attribute("file.total_bytes_written", self._bytes_written)
                self._parent_span.end()
            return self._file.close()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()
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
    if db_libraries is None or "sqlalchemy" in db_libraries:
        try:
            from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

            SQLAlchemyInstrumentor().instrument()
            instrumented.append("sqlalchemy")
            logger.info("Instrumented: sqlalchemy")
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

    try:
        content_type = response.headers.get("content-type", "")
        if _is_text_content_type(content_type):
            _span_processor.store_body(span.context.span_id, response_body=response.text)
    except Exception:
        pass


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

    try:
        # Capture request headers from RequestInfo namedtuple
        if hasattr(request, 'headers') and request.headers:
            request_headers = dict(request.headers)
            _span_processor.store_body(span.context.span_id, request_headers=request_headers)

        # Try to get request body - RequestInfo has a 'stream' attribute
        body = None
        if hasattr(request, 'stream'):
            stream = request.stream
            if hasattr(stream, 'body'):
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

    try:
        # Capture response headers first (always available even for streaming)
        if hasattr(response, 'headers') and response.headers:
            response_headers = dict(response.headers)
            _span_processor.store_body(span.context.span_id, response_headers=response_headers)

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
                _span_processor.store_body(span.context.span_id, response_body=body)
    except Exception:
        pass  # Best effort


async def _httpx_async_request_hook(span, request) -> None:
    """Async version of request hook."""
    _httpx_request_hook(span, request)


async def _httpx_async_response_hook(span, request, response) -> None:
    """Async version of response hook."""
    if _span_processor is None:
        return

    # Check if URL should be ignored
    url = str(request.url) if hasattr(request, 'url') else None
    if url and _should_ignore_url(url):
        return

    try:
        # Capture response headers
        if hasattr(response, 'headers') and response.headers:
            response_headers = dict(response.headers)
            _span_processor.store_body(span.context.span_id, response_headers=response_headers)

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
                _span_processor.store_body(span.context.span_id, response_body=body)

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


# ═══════════════════════════════════════════════════════════════════════════════
# httpx body capture (patches Client.send)
# ═══════════════════════════════════════════════════════════════════════════════


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
            # Check if URL should be ignored
            url = str(request.url) if hasattr(request, 'url') else None
            if url and _should_ignore_url(url):
                return _original_send(self, request, *args, **kwargs)

            # Capture request body BEFORE sending
            request_body = None
            try:
                if hasattr(request, '_content') and request._content:
                    request_body = request._content
                    if isinstance(request_body, bytes):
                        request_body = request_body.decode("utf-8", errors="ignore")
                elif hasattr(request, 'content') and request.content:
                    request_body = request.content
                    if isinstance(request_body, bytes):
                        request_body = request_body.decode("utf-8", errors="ignore")
            except Exception as e:
                logger.debug(f"Failed to capture request body: {e}")

            response = _original_send(self, request, *args, **kwargs)

            # Capture response body AFTER receiving (skip binary)
            response_body = None
            content_type = response.headers.get("content-type", "")
            if _is_text_content_type(content_type):
                try:
                    response_body = response.text
                except (UnicodeDecodeError, Exception) as e:
                    logger.debug(f"Failed to capture response body: {e}")

            # Store bodies if we have an active span
            try:
                from opentelemetry import trace

                span = trace.get_current_span()
                if span and hasattr(span, 'context') and span.context.span_id:
                    if request_body:
                        span_processor.store_body(span.context.span_id, request_body=request_body)
                        logger.debug(f"Stored request body for span {span.context.span_id}")
                    if response_body:
                        span_processor.store_body(span.context.span_id, response_body=response_body)
                        logger.debug(f"Stored response body for span {span.context.span_id}")
            except Exception as e:
                logger.debug(f"Failed to store body: {e}")

            return response

        async def _patched_async_send(self, request, *args, **kwargs):
            # Check if URL should be ignored
            url = str(request.url) if hasattr(request, 'url') else None
            if url and _should_ignore_url(url):
                return await _original_async_send(self, request, *args, **kwargs)

            # Capture request body and headers BEFORE sending
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
                # Capture request headers
                if hasattr(request, 'headers') and request.headers:
                    request_headers = dict(request.headers)
            except Exception as e:
                logger.debug(f"Failed to capture request body/headers: {e}")

            # Get current span BEFORE calling original send
            # The OTel httpx instrumentation creates a child span for HTTP call
            from opentelemetry import trace
            parent_span = trace.get_current_span()

            response = await _original_async_send(self, request, *args, **kwargs)

            # Capture response body and headers AFTER receiving (skip binary for body)
            response_body = None
            response_headers = None
            content_type = response.headers.get("content-type", "")
            try:
                # Always capture response headers
                if hasattr(response, 'headers') and response.headers:
                    response_headers = dict(response.headers)
                # Only capture body for text content types
                if _is_text_content_type(content_type):
                    response_body = response.text
            except (UnicodeDecodeError, Exception) as e:
                logger.debug(f"Failed to capture response body: {e}")

            # Store bodies and headers against parent span (activity span)
            # The HTTP span may have ended by now, but we stored it via hooks
            try:
                if parent_span and hasattr(parent_span, 'context') and parent_span.context.span_id:
                    span_id = parent_span.context.span_id
                    if request_body:
                        span_processor.store_body(span_id, request_body=request_body)
                    if response_body:
                        span_processor.store_body(span_id, response_body=response_body)
                    if request_headers:
                        span_processor.store_body(span_id, request_headers=request_headers)
                    if response_headers:
                        span_processor.store_body(span_id, response_headers=response_headers)
            except Exception:
                pass  # Best effort

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

    try:
        if hasattr(request_info, "body") and request_info.body:
            body = request_info.body
            if isinstance(body, bytes):
                body = body.decode("utf-8", errors="ignore")
            _span_processor.store_body(span.context.span_id, request_body=body)
    except Exception:
        pass


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

    try:
        content_type = response.headers.get("content-type", "")
        if _is_text_content_type(content_type):
            body = response.data
            if isinstance(body, bytes):
                body = body.decode("utf-8", errors="ignore")
            if body:
                _span_processor.store_body(span.context.span_id, response_body=body)
    except Exception:
        pass


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