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
- sqlite3 (stdlib)
- pymongo (MongoDB)
- redis
- sqlalchemy (ORM)
"""

from typing import TYPE_CHECKING, Any, List, Optional, Set
import logging

if TYPE_CHECKING:
    from .span_processor import WorkflowSpanProcessor

logger = logging.getLogger(__name__)

# Global state — hooks in sub-modules reference these via late import of this module
_span_processor: Optional["WorkflowSpanProcessor"] = None
_ignored_url_prefixes: Set[str] = set()

# Hook-level governance is handled by hook_governance module
from . import hook_governance as _hook_gov
from . import db_governance_hooks as _db_gov

# ── Re-export all names from sub-modules for backward compatibility ─────────
# Tests and external code import from openbox.otel_setup; these re-exports
# ensure all existing import paths continue to work unchanged.
from .http_governance_hooks import (  # noqa: F401
    _should_ignore_url,
    _is_text_content_type,
    _build_http_span_data,
    _requests_request_hook,
    _requests_response_hook,
    _httpx_request_hook,
    _httpx_response_hook,
    _httpx_async_request_hook,
    _httpx_async_response_hook,
    _capture_httpx_request_data,
    _capture_httpx_response_data,
    _get_httpx_http_span,
    _prepare_completed_governance,
    setup_httpx_body_capture,
    _urllib3_request_hook,
    _urllib3_response_hook,
    _urllib_request_hook,
    _httpx_http_span,
    _http_hook_timings,
    _HTTP_HOOK_TIMINGS_MAX,
    _TEXT_CONTENT_TYPES,
)
from .file_governance_hooks import (  # noqa: F401
    _build_file_span_data,
    setup_file_io_instrumentation,
    uninstrument_file_io,
)


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
    max_body_size: Optional[int] = None,
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
                      "sqlite3", "pymongo", "redis", "sqlalchemy"
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

    # Configure governance modules
    _hook_gov.configure(
        api_url,
        api_key,
        span_processor,
        api_timeout=api_timeout,
        on_api_error=on_api_error,
        max_body_size=max_body_size,
    )
    _db_gov.configure(span_processor)

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

    logger.info(
        f"OpenTelemetry HTTP instrumentation complete. Instrumented: {instrumented}"
    )

    # 6. Database instrumentation (optional)
    if sqlalchemy_engine is not None and not instrument_databases:
        logger.warning(
            "sqlalchemy_engine was provided but instrument_databases=False; "
            "engine will not be instrumented"
        )
    if instrument_databases:
        db_instrumented = setup_database_instrumentation(
            db_libraries, sqlalchemy_engine
        )
        if db_instrumented:
            instrumented.extend(db_instrumented)

    # 7. File I/O instrumentation (optional)
    if instrument_file_io and setup_file_io_instrumentation():
        instrumented.append("file_io")

    # 8. Context propagation for async activities using run_in_executor
    # Without this, OTel trace context is lost in executor threads and
    # hook governance silently skips HTTP/DB/file spans from those threads.
    from .context_propagation import install_context_propagating_executor

    install_context_propagating_executor()

    logger.info(
        f"OpenTelemetry governance setup complete. Instrumented: {instrumented}"
    )


def _instrument_sqlalchemy(
    db_libraries: Optional[Set[str]],
    sqlalchemy_engine: Optional[Any],
    instrumented: List[str],
) -> None:
    """Handle sqlalchemy instrumentation with optional engine validation."""
    if (
        sqlalchemy_engine is not None
        and db_libraries is not None
        and "sqlalchemy" not in db_libraries
    ):
        logger.warning(
            "sqlalchemy_engine provided but 'sqlalchemy' not in db_libraries; skipping"
        )
        return

    if db_libraries is not None and "sqlalchemy" not in db_libraries:
        return

    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        if sqlalchemy_engine is not None:
            _validate_sqlalchemy_engine(sqlalchemy_engine)
            _db_gov.setup_sqlalchemy_hooks(sqlalchemy_engine)
            SQLAlchemyInstrumentor().instrument(engine=sqlalchemy_engine)
            logger.info("Instrumented: sqlalchemy (existing engine)")
        else:
            SQLAlchemyInstrumentor().instrument()
            logger.info("Instrumented: sqlalchemy (future engines)")
        instrumented.append("sqlalchemy")
    except ImportError:
        logger.debug("sqlalchemy instrumentation not available")


def _validate_sqlalchemy_engine(engine: Any) -> None:
    """Validate that engine is a real SQLAlchemy Engine instance."""
    try:
        from sqlalchemy.engine import Engine as _SAEngine
    except ImportError:
        raise TypeError("sqlalchemy_engine provided but sqlalchemy is not installed")
    if not isinstance(engine, _SAEngine):
        raise TypeError(
            f"sqlalchemy_engine must be a sqlalchemy.engine.Engine instance, "
            f"got {type(engine).__name__}"
        )


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
                      - "sqlite3" (stdlib)
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

    # pymongo CommandListener must register before MongoClient creation
    if db_libraries is None or "pymongo" in db_libraries:
        _db_gov.setup_pymongo_hooks()

    # Standard OTel instrumentors (governance via CursorTracer patch below)
    _INSTRUMENTORS = [
        ("psycopg2", "opentelemetry.instrumentation.psycopg2", "Psycopg2Instrumentor"),
        ("asyncpg", "opentelemetry.instrumentation.asyncpg", "AsyncPGInstrumentor"),
        ("mysql", "opentelemetry.instrumentation.mysql", "MySQLInstrumentor"),
        ("pymysql", "opentelemetry.instrumentation.pymysql", "PyMySQLInstrumentor"),
        ("sqlite3", "opentelemetry.instrumentation.sqlite3", "SQLite3Instrumentor"),
        ("pymongo", "opentelemetry.instrumentation.pymongo", "PymongoInstrumentor"),
    ]
    for lib_name, module_path, class_name in _INSTRUMENTORS:
        if db_libraries is not None and lib_name not in db_libraries:
            continue
        try:
            mod = __import__(module_path, fromlist=[class_name])
            getattr(mod, class_name)().instrument()
            instrumented.append(lib_name)
            logger.info(f"Instrumented: {lib_name}")
        except (ImportError, Exception):
            logger.debug(f"{lib_name} OTel instrumentation not available")

    # redis — pass governance hooks to OTel instrumentor (native hook support)
    if db_libraries is None or "redis" in db_libraries:
        try:
            from opentelemetry.instrumentation.redis import RedisInstrumentor
            req_hook, resp_hook = _db_gov.setup_redis_hooks()
            RedisInstrumentor().instrument(request_hook=req_hook, response_hook=resp_hook)
            instrumented.append("redis")
            logger.info("Instrumented: redis")
        except Exception:
            logger.debug("redis instrumentation not available")

    # sqlalchemy — special handling for existing engines
    _instrument_sqlalchemy(db_libraries, sqlalchemy_engine, instrumented)

    # ── Governance hooks for dbapi libs (must be AFTER instrumentors) ──
    # OTel dbapi instrumentors silently discard request_hook/response_hook kwargs.
    # Instead, we patch CursorTracer.traced_execution to inject governance hooks
    # around the query_method call (runs inside the OTel span context).
    dbapi_libs = {"psycopg2", "mysql", "pymysql", "sqlite3"}
    if any(lib in instrumented for lib in dbapi_libs):
        if _db_gov.install_cursor_tracer_hooks():
            logger.info("CursorTracer governance hooks installed for dbapi libs")

    # asyncpg uses its own _do_execute (not CursorTracer) — needs separate wrapt hooks
    if "asyncpg" in instrumented:
        _db_gov.install_asyncpg_hooks()

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
    except Exception:
        pass

    try:
        from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor

        AsyncPGInstrumentor().uninstrument()
    except Exception:
        pass

    try:
        from opentelemetry.instrumentation.mysql import MySQLInstrumentor

        MySQLInstrumentor().uninstrument()
    except Exception:
        pass

    try:
        from opentelemetry.instrumentation.pymysql import PyMySQLInstrumentor

        PyMySQLInstrumentor().uninstrument()
    except Exception:
        pass

    try:
        from opentelemetry.instrumentation.sqlite3 import SQLite3Instrumentor

        SQLite3Instrumentor().uninstrument()
    except Exception:
        pass

    try:
        from opentelemetry.instrumentation.pymongo import PymongoInstrumentor

        PymongoInstrumentor().uninstrument()
    except Exception:
        pass

    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor

        RedisInstrumentor().uninstrument()
    except Exception:
        pass

    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        SQLAlchemyInstrumentor().uninstrument()
    except Exception:
        pass

    # Clean up DB governance hooks
    _db_gov.uninstrument_all()


def uninstrument_all() -> None:
    """Uninstrument all HTTP and database libraries."""
    global _span_processor, _ignored_url_prefixes
    _span_processor = None
    _ignored_url_prefixes = set()

    # Uninstrument HTTP libraries
    try:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor

        RequestsInstrumentor().uninstrument()
    except Exception:
        pass

    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().uninstrument()
    except Exception:
        pass

    try:
        from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor

        URLLib3Instrumentor().uninstrument()
    except Exception:
        pass

    try:
        from opentelemetry.instrumentation.urllib import URLLibInstrumentor

        URLLibInstrumentor().uninstrument()
    except Exception:
        pass

    # Uninstrument database libraries
    uninstrument_databases()

    # Uninstrument file I/O
    uninstrument_file_io()
