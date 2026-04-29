# openbox/db_governance_hooks.py
"""Hook-level governance for database operations.

Intercepts DB queries at 'started' (pre-query) and 'completed' (post-query)
stages, sending governance evaluations to OpenBox Core via hook_governance.

Supported libraries:
- All dbapi-based (psycopg2, asyncpg, mysql, pymysql) via CursorTracer patch
- pymongo (CommandListener monitoring API)
- redis (native OTel request_hook/response_hook)
- sqlalchemy (before/after_cursor_execute events)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from opentelemetry import trace as otel_trace

if TYPE_CHECKING:
    from .span_processor import WorkflowSpanProcessor

logger = logging.getLogger(__name__)

# Track installed wrapt patches (informational — wrapt patches can't be cleanly removed)
_installed_patches: List[Tuple[str, str]] = []

# Track SQLAlchemy event listeners for cleanup: (engine, event_name, listener_fn)
_sqlalchemy_listeners: List[Tuple[Any, str, Callable]] = []

# pymongo dedup: thread-local depth counter for wrapt wrapper nesting.
_pymongo_wrapt_depth = threading.local()

# pymongo: store command string from started event (keyed by request_id)
_pymongo_pending_commands: Dict[int, str] = {}
_PYMONGO_PENDING_MAX = 1000

_span_processor: Optional["WorkflowSpanProcessor"] = None


def configure(span_processor: "WorkflowSpanProcessor") -> None:
    """Store span_processor reference for span data building."""
    global _span_processor
    _span_processor = span_processor
    logger.info("DB governance hooks configured")


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _classify_sql(query: Any) -> str:
    """Extract SQL verb from a query string."""
    if not query:
        return "UNKNOWN"
    q = str(query).strip().upper()
    for verb in (
        "SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP",
        "ALTER", "TRUNCATE", "BEGIN", "COMMIT", "ROLLBACK", "EXPLAIN",
    ):
        if q.startswith(verb):
            return verb
    return "UNKNOWN"


def _generate_span_id() -> str:
    """Generate a random 16-hex-char span ID for pymongo governance spans.

    Uses secrets (CSPRNG) rather than random — span IDs are not secrets, but
    the stronger generator silences security scanners without runtime cost
    and removes any chance of cross-trace ID collision under heavy load.
    """
    import secrets
    return secrets.token_hex(8)


def _build_db_span_data(
    span: Any,
    db_system: str,
    db_name: Optional[str],
    db_operation: str,
    db_statement: str,
    server_address: Optional[str],
    server_port: Optional[int],
    stage: str,
    duration_ms: Optional[float] = None,
    error: Optional[str] = None,
    rowcount: Optional[int] = None,
    gov_span_id: Optional[str] = None,
) -> dict:
    """Build span data dict for a DB operation."""
    from . import hook_governance as _hook_gov

    current_span_id, trace_id_hex, default_parent = _hook_gov.extract_span_context(span)

    if gov_span_id:
        span_id_hex = gov_span_id
        parent_span_id = current_span_id
    else:
        span_id_hex = current_span_id
        parent_span_id = default_parent

    raw_attrs = getattr(span, "attributes", None)
    attrs = dict(raw_attrs) if raw_attrs and isinstance(raw_attrs, dict) else {}

    span_name = getattr(span, "name", None)
    if not span_name or not isinstance(span_name, str):
        span_name = f"{db_operation} {db_system}"
    now_ns = time.time_ns()

    return {
        "span_id": span_id_hex,
        "trace_id": trace_id_hex,
        "parent_span_id": parent_span_id,
        "name": span_name,
        "kind": "CLIENT",
        "stage": stage,
        "start_time": now_ns,
        "end_time": now_ns if stage == "completed" else None,
        "duration_ns": int(duration_ms * 1_000_000) if duration_ms else None,
        "attributes": attrs,
        "status": {"code": "ERROR" if error else "UNSET", "description": error},
        "events": [],
        "hook_type": "db_query",
        "db_system": db_system,
        "db_name": str(db_name) if db_name else None,
        "db_operation": db_operation,
        "db_statement": db_statement,
        "server_address": server_address,
        "server_port": int(server_port) if server_port else None,
        "rowcount": (
            rowcount
            if rowcount is not None and isinstance(rowcount, int) and rowcount >= 0
            else None
        ),
        "error": error,
    }


def _db_identifier(
    db_system: str, server_address: Optional[str],
    server_port: Optional[int], db_name: Optional[str],
) -> str:
    """Build a stable identifier string for DB governance evaluations."""
    return f"{db_system}://{server_address or 'unknown'}:{server_port or 0}/{db_name or ''}"


def _get_rowcount(cursor) -> Optional[int]:
    """Safely extract rowcount from a cursor."""
    try:
        rc = getattr(cursor, "rowcount", -1)
        if rc is not None and rc >= 0:
            return rc
    except Exception:
        pass
    return None


def _evaluate_db_sync(
    identifier: str, span_data: dict, *, is_completed: bool = False,
) -> None:
    """Send DB governance evaluation (sync)."""
    from . import hook_governance as _hook_gov

    if not _hook_gov.is_configured():
        return
    span = otel_trace.get_current_span()
    if is_completed:
        try:
            _hook_gov.evaluate_sync(span, identifier=identifier, span_data=span_data)
        except Exception as e:
            logger.debug(f"DB governance completed evaluation error (non-blocking): {e}")
    else:
        _hook_gov.evaluate_sync(span, identifier=identifier, span_data=span_data)


async def _evaluate_db_async(
    identifier: str, span_data: dict, *, is_completed: bool = False,
) -> None:
    """Send DB governance evaluation (async)."""
    from . import hook_governance as _hook_gov

    if not _hook_gov.is_configured():
        return
    span = otel_trace.get_current_span()
    if is_completed:
        try:
            await _hook_gov.evaluate_async(span, identifier=identifier, span_data=span_data)
        except Exception as e:
            logger.debug(f"DB governance completed evaluation error (non-blocking): {e}")
    else:
        await _hook_gov.evaluate_async(span, identifier=identifier, span_data=span_data)


# ─── Governed query execution (shared by CursorTracer and pymongo wrapt) ──────


def _run_governed_query_sync(
    query_method, args, kwargs, *,
    db_system: str, db_name: Optional[str], operation: str, stmt: str,
    host: Optional[str], port: Optional[int],
    cursor=None, gov_span_id: Optional[str] = None,
):
    """Execute a DB query with sync governance (started + completed)."""
    from .types import GovernanceBlockedError

    current_span = otel_trace.get_current_span()
    ident = _db_identifier(db_system, host, port, db_name)

    started_sd = _build_db_span_data(
        current_span, db_system, db_name, operation, stmt, host, port,
        "started", gov_span_id=gov_span_id,
    )
    _evaluate_db_sync(ident, started_sd)

    start = time.perf_counter()
    try:
        result = query_method(*args, **kwargs)
        duration_ms = (time.perf_counter() - start) * 1000
        rc = _get_rowcount(cursor) if cursor else None
        completed_sd = _build_db_span_data(
            current_span, db_system, db_name, operation, stmt, host, port,
            "completed", duration_ms=duration_ms, rowcount=rc, gov_span_id=gov_span_id,
        )
        _evaluate_db_sync(ident, completed_sd, is_completed=True)
        return result
    except GovernanceBlockedError:
        raise
    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000
        completed_sd = _build_db_span_data(
            current_span, db_system, db_name, operation, stmt, host, port,
            "completed", duration_ms=duration_ms, error=str(e), gov_span_id=gov_span_id,
        )
        _evaluate_db_sync(ident, completed_sd, is_completed=True)
        raise


async def _run_governed_query_async(
    query_method, args, kwargs, *,
    db_system: str, db_name: Optional[str], operation: str, stmt: str,
    host: Optional[str], port: Optional[int],
    cursor=None,
):
    """Execute a DB query with async governance (started + completed)."""
    from .types import GovernanceBlockedError

    current_span = otel_trace.get_current_span()
    ident = _db_identifier(db_system, host, port, db_name)

    started_sd = _build_db_span_data(
        current_span, db_system, db_name, operation, stmt, host, port, "started",
    )
    await _evaluate_db_async(ident, started_sd)

    start = time.perf_counter()
    try:
        result = await query_method(*args, **kwargs)
        duration_ms = (time.perf_counter() - start) * 1000
        rc = _get_rowcount(cursor) if cursor else None
        completed_sd = _build_db_span_data(
            current_span, db_system, db_name, operation, stmt, host, port,
            "completed", duration_ms=duration_ms, rowcount=rc,
        )
        await _evaluate_db_async(ident, completed_sd, is_completed=True)
        return result
    except GovernanceBlockedError:
        raise
    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000
        completed_sd = _build_db_span_data(
            current_span, db_system, db_name, operation, stmt, host, port,
            "completed", duration_ms=duration_ms, error=str(e),
        )
        await _evaluate_db_async(ident, completed_sd, is_completed=True)
        raise


def _extract_dbapi_context(tracer_self, args):
    """Extract DB metadata from CursorTracer instance."""
    db_system = tracer_self._db_api_integration.database_system
    db_name = tracer_self._db_api_integration.database
    query = args[0] if args else ""
    operation = _classify_sql(query)
    stmt = str(query)[:2000]
    host = tracer_self._db_api_integration.connection_props.get("host", "unknown")
    port = tracer_self._db_api_integration.connection_props.get("port")
    return db_system, db_name, operation, stmt, host, port


# ═══════════════════════════════════════════════════════════════════════════════
# CursorTracer patch
# ═══════════════════════════════════════════════════════════════════════════════

_orig_traced_execution: Optional[Callable] = None
_orig_traced_execution_async: Optional[Callable] = None


def install_cursor_tracer_hooks() -> bool:
    """Monkey-patch OTel CursorTracer to inject governance hooks."""
    global _orig_traced_execution, _orig_traced_execution_async

    try:
        from opentelemetry.instrumentation.dbapi import CursorTracer
    except ImportError:
        logger.debug("OTel dbapi not available for CursorTracer patching")
        return False

    if _orig_traced_execution is not None:
        logger.debug("CursorTracer already patched — skipping")
        return True

    _orig_traced_execution = CursorTracer.traced_execution
    _orig_traced_execution_async = CursorTracer.traced_execution_async

    def _gov_traced_execution(self, cursor, query_method, *args, **kwargs):
        db_system, db_name, operation, stmt, host, port = _extract_dbapi_context(self, args)

        def _governed_query(*qargs, **qkwargs):
            return _run_governed_query_sync(
                query_method, qargs, qkwargs,
                db_system=db_system, db_name=db_name, operation=operation,
                stmt=stmt, host=host, port=port, cursor=cursor,
            )

        return _orig_traced_execution(self, cursor, _governed_query, *args, **kwargs)

    async def _gov_traced_execution_async(self, cursor, query_method, *args, **kwargs):
        db_system, db_name, operation, stmt, host, port = _extract_dbapi_context(self, args)

        async def _governed_query_async(*qargs, **qkwargs):
            return await _run_governed_query_async(
                query_method, qargs, qkwargs,
                db_system=db_system, db_name=db_name, operation=operation,
                stmt=stmt, host=host, port=port, cursor=cursor,
            )

        return await _orig_traced_execution_async(
            self, cursor, _governed_query_async, *args, **kwargs
        )

    CursorTracer.traced_execution = _gov_traced_execution
    CursorTracer.traced_execution_async = _gov_traced_execution_async
    logger.info("CursorTracer patched with governance hooks (all dbapi libs)")
    return True


def _uninstall_cursor_tracer_hooks() -> None:
    """Restore original CursorTracer methods."""
    global _orig_traced_execution, _orig_traced_execution_async

    if _orig_traced_execution is None:
        return

    try:
        from opentelemetry.instrumentation.dbapi import CursorTracer
        CursorTracer.traced_execution = _orig_traced_execution
        CursorTracer.traced_execution_async = _orig_traced_execution_async
    except ImportError:
        pass

    _orig_traced_execution = None
    _orig_traced_execution_async = None
    logger.debug("CursorTracer governance hooks removed")


# ═══════════════════════════════════════════════════════════════════════════════
# asyncpg — wrapt wrapper AFTER OTel
# ═══════════════════════════════════════════════════════════════════════════════

_asyncpg_patched = False


def _extract_asyncpg_context(instance):
    """Extract connection metadata from asyncpg Connection."""
    params = getattr(instance, "_params", None)
    host = (
        getattr(instance, "_addr", ("unknown",))[0]
        if hasattr(instance, "_addr") else "unknown"
    )
    port = (
        getattr(instance, "_addr", (None, 5432))[1]
        if hasattr(instance, "_addr") else 5432
    )
    db_name = getattr(params, "database", None) if params else None
    return host, port, db_name


def install_asyncpg_hooks() -> bool:
    """Install governance hooks on asyncpg via wrapt wrapping."""
    global _asyncpg_patched
    if _asyncpg_patched:
        return True

    try:
        import wrapt
        import asyncpg  # noqa: F401
    except ImportError:
        logger.debug("asyncpg or wrapt not available for governance hooks")
        return False

    async def _asyncpg_governance_wrapper(wrapped, instance, args, kwargs):
        query = args[0] if args else ""
        operation = _classify_sql(query)
        stmt = str(query)[:2000]
        host, port, db_name = _extract_asyncpg_context(instance)

        return await _run_governed_query_async(
            wrapped, args, kwargs,
            db_system="postgresql", db_name=db_name, operation=operation,
            stmt=stmt, host=host, port=port,
        )

    methods = [
        ("asyncpg.connection", "Connection.execute"),
        ("asyncpg.connection", "Connection.executemany"),
        ("asyncpg.connection", "Connection.fetch"),
        ("asyncpg.connection", "Connection.fetchval"),
        ("asyncpg.connection", "Connection.fetchrow"),
    ]
    patched = 0
    for module, method in methods:
        try:
            wrapt.wrap_function_wrapper(module, method, _asyncpg_governance_wrapper)
            _installed_patches.append((module, method))
            patched += 1
        except (AttributeError, TypeError, ImportError) as e:
            logger.debug(f"asyncpg governance hook failed for {method}: {e}")

    if patched > 0:
        _asyncpg_patched = True
        logger.info(f"asyncpg governance hooks installed: {patched}/{len(methods)} methods")
        return True

    logger.debug("No asyncpg methods patched for governance")
    return False


def _uninstall_asyncpg_hooks() -> None:
    """Remove asyncpg wrapt governance hooks."""
    global _asyncpg_patched
    if not _asyncpg_patched:
        return
    _asyncpg_patched = False


# ═══════════════════════════════════════════════════════════════════════════════
# pymongo (CommandListener + wrapt)
# ═══════════════════════════════════════════════════════════════════════════════

_pymongo_listener: Any = None


def setup_pymongo_hooks() -> None:
    """Install governance hooks on pymongo via monitoring.CommandListener."""
    global _pymongo_listener
    try:
        import pymongo.monitoring

        class _GovernanceCommandListener(pymongo.monitoring.CommandListener):
            """Pymongo CommandListener that sends governance evaluations."""

            def started(self, event):
                if getattr(_pymongo_wrapt_depth, "value", 0) > 0:
                    return
                try:
                    span = otel_trace.get_current_span()
                    host, port = _extract_pymongo_address(event)
                    cmd_str = str(event.command)[:2000]
                    if len(_pymongo_pending_commands) >= _PYMONGO_PENDING_MAX:
                        _pymongo_pending_commands.clear()
                    _pymongo_pending_commands[event.request_id] = cmd_str
                    started_sd = _build_db_span_data(
                        span, "mongodb", event.database_name, event.command_name,
                        cmd_str, host, port, "started",
                    )
                    ident = _db_identifier("mongodb", host, port, event.database_name)
                    _evaluate_db_sync(ident, started_sd)
                except Exception as e:
                    logger.debug(f"pymongo governance started error: {e}")

            def succeeded(self, event):
                if getattr(_pymongo_wrapt_depth, "value", 0) > 0:
                    _pymongo_pending_commands.pop(event.request_id, None)
                    return
                self._send_completed(event)

            def failed(self, event):
                if getattr(_pymongo_wrapt_depth, "value", 0) > 0:
                    _pymongo_pending_commands.pop(event.request_id, None)
                    return
                self._send_completed(event, error=str(event.failure))

            def _send_completed(self, event, error=None):
                try:
                    span = otel_trace.get_current_span()
                    host, port = _extract_pymongo_address(event)
                    duration_ms = event.duration_micros / 1000.0
                    cmd_str = _pymongo_pending_commands.pop(
                        event.request_id, event.command_name
                    )
                    completed_sd = _build_db_span_data(
                        span, "mongodb", event.database_name, event.command_name,
                        cmd_str, host, port, "completed",
                        duration_ms=duration_ms, error=error,
                    )
                    ident = _db_identifier("mongodb", host, port, event.database_name)
                    _evaluate_db_sync(ident, completed_sd, is_completed=True)
                except Exception as e:
                    logger.debug(f"pymongo governance completed error: {e}")

        _pymongo_listener = _GovernanceCommandListener()
        pymongo.monitoring.register(_pymongo_listener)
        logger.info("DB governance hooks installed: pymongo (CommandListener)")
    except ImportError:
        logger.debug("pymongo not available for governance hooks")

    _setup_pymongo_wrapt_hooks()


def _extract_pymongo_address(event) -> Tuple[str, int]:
    """Extract (host, port) from a pymongo monitoring event."""
    try:
        addr = event.connection_id
        if addr and len(addr) >= 2:
            return str(addr[0]), int(addr[1])
    except (AttributeError, TypeError, IndexError):
        pass
    return "unknown", 27017


def _extract_pymongo_collection_context(instance, wrapped):
    """Extract DB metadata from a pymongo Collection wrapper call."""
    db_name = instance.database.name
    operation = wrapped.__name__
    try:
        address = instance.database.client.address
        host, port = address[0], address[1]
    except (AttributeError, TypeError):
        host, port = "unknown", 27017
    statement = f"{instance.name}.{operation}"
    return db_name, operation, host, port, statement


def _setup_pymongo_wrapt_hooks() -> None:
    """Best-effort wrapt wrapping of pymongo Collection methods for blocking."""
    try:
        import wrapt

        def _collection_wrapper(wrapped, instance, args, kwargs):
            depth = getattr(_pymongo_wrapt_depth, "value", 0)
            _pymongo_wrapt_depth.value = depth + 1

            # Nested call — just pass through
            if depth > 0:
                try:
                    return wrapped(*args, **kwargs)
                finally:
                    _pymongo_wrapt_depth.value = getattr(_pymongo_wrapt_depth, "value", 1) - 1

            # Outermost call — fire governance
            db_name, operation, host, port, statement = (
                _extract_pymongo_collection_context(instance, wrapped)
            )
            gov_sid = _generate_span_id()
            try:
                return _run_governed_query_sync(
                    wrapped, args, kwargs,
                    db_system="mongodb", db_name=db_name, operation=operation,
                    stmt=statement, host=host, port=port, gov_span_id=gov_sid,
                )
            finally:
                _pymongo_wrapt_depth.value = getattr(_pymongo_wrapt_depth, "value", 1) - 1

        methods = (
            "find", "find_one", "insert_one", "insert_many",
            "update_one", "update_many", "delete_one", "delete_many",
            "aggregate", "count_documents",
        )
        patched = 0
        for method in methods:
            try:
                wrapt.wrap_function_wrapper(
                    "pymongo.collection", f"Collection.{method}", _collection_wrapper
                )
                _installed_patches.append(("pymongo.collection", f"Collection.{method}"))
                patched += 1
            except (AttributeError, TypeError):
                pass
        if patched > 0:
            logger.info(f"pymongo wrapt hooks installed: {patched}/{len(methods)} methods")
        else:
            logger.debug("pymongo Collection wrapt hooks failed (C extension or immutable)")
    except ImportError:
        logger.debug("wrapt not available for pymongo blocking hooks")


# ═══════════════════════════════════════════════════════════════════════════════
# redis (native OTel hooks)
# ═══════════════════════════════════════════════════════════════════════════════

_redis_span_meta: Dict[int, Tuple[float, str, str, str, int, str]] = {}
_REDIS_META_MAX = 1000


def setup_redis_hooks() -> Tuple[Callable, Callable]:
    """Return (request_hook, response_hook) for RedisInstrumentor."""

    def _request_hook(span, instance, args, kwargs):
        command = str(args[0]) if args else "UNKNOWN"
        statement = " ".join(str(a) for a in args) if args else ""
        try:
            conn_kwargs = instance.connection_pool.connection_kwargs
            host = conn_kwargs.get("host", "localhost")
            port = conn_kwargs.get("port", 6379)
            db_name = str(conn_kwargs.get("db", 0))
        except AttributeError:
            host, port, db_name = "localhost", 6379, "0"

        ident = _db_identifier("redis", host, port, db_name)
        started_sd = _build_db_span_data(
            span, "redis", db_name, command, statement, host, port, "started"
        )
        _evaluate_db_sync(ident, started_sd)
        if len(_redis_span_meta) >= _REDIS_META_MAX:
            _redis_span_meta.clear()
        _redis_span_meta[id(span)] = (
            time.perf_counter(), command, statement, host, port, db_name,
        )

    def _response_hook(span, instance, response):
        meta = _redis_span_meta.pop(id(span), None)
        start_time = meta[0] if meta else time.perf_counter()
        command = meta[1] if meta else "UNKNOWN"
        statement = meta[2] if meta else ""
        host = meta[3] if meta and len(meta) > 3 else "localhost"
        port = meta[4] if meta and len(meta) > 4 else 6379
        db_name = meta[5] if meta and len(meta) > 5 else "0"
        duration_ms = (time.perf_counter() - start_time) * 1000

        ident = _db_identifier("redis", host, port, db_name)
        completed_sd = _build_db_span_data(
            span, "redis", db_name, command, statement, host, port,
            "completed", duration_ms=duration_ms,
        )
        _evaluate_db_sync(ident, completed_sd, is_completed=True)

    return _request_hook, _response_hook


# ═══════════════════════════════════════════════════════════════════════════════
# sqlalchemy (native before/after_cursor_execute events)
# ═══════════════════════════════════════════════════════════════════════════════

_sa_timings: Dict[Tuple[int, int], float] = {}
_SA_TIMINGS_MAX = 1000


def _get_sa_db_system(engine) -> str:
    """Extract db_system from SQLAlchemy engine dialect name."""
    dialect = getattr(engine, "dialect", None)
    name = getattr(dialect, "name", "") if dialect else ""
    mapping = {
        "postgresql": "postgresql", "mysql": "mysql", "sqlite": "sqlite",
        "oracle": "oracle", "mssql": "mssql",
    }
    return mapping.get(name, name or "unknown")


def setup_sqlalchemy_hooks(engine) -> None:
    """Register SQLAlchemy event listeners for governance on the given engine."""
    try:
        from sqlalchemy import event
        from sqlalchemy.engine import Engine as _SAEngine
    except ImportError:
        logger.debug("sqlalchemy not available for governance hooks")
        return

    if not isinstance(engine, _SAEngine):
        logger.debug("Skipping SQLAlchemy governance hooks: not a real Engine instance")
        return

    def _before_execute(conn, cursor, statement, parameters, context, executemany):
        if len(_sa_timings) >= _SA_TIMINGS_MAX:
            _sa_timings.clear()
        _sa_timings[(id(conn), id(cursor))] = time.perf_counter()
        db_system = _get_sa_db_system(conn.engine)
        db_name = conn.engine.url.database
        operation = _classify_sql(statement)
        host = conn.engine.url.host
        port = conn.engine.url.port

        current_span = otel_trace.get_current_span()
        ident = _db_identifier(db_system, host, port, db_name)
        started_sd = _build_db_span_data(
            current_span, db_system, db_name, operation, str(statement),
            host, port, "started",
        )
        _evaluate_db_sync(ident, started_sd)

    def _after_execute(conn, cursor, statement, parameters, context, executemany):
        start = _sa_timings.pop((id(conn), id(cursor)), None)
        duration_ms = (time.perf_counter() - start) * 1000 if start else 0.0
        db_system = _get_sa_db_system(conn.engine)
        db_name = conn.engine.url.database
        operation = _classify_sql(statement)
        host = conn.engine.url.host
        port = conn.engine.url.port

        current_span = otel_trace.get_current_span()
        ident = _db_identifier(db_system, host, port, db_name)
        completed_sd = _build_db_span_data(
            current_span, db_system, db_name, operation, str(statement),
            host, port, "completed", duration_ms=duration_ms,
        )
        _evaluate_db_sync(ident, completed_sd, is_completed=True)

    def _on_error(context):
        """Handle DB errors — clean up timing and send completed with error."""
        cursor = getattr(context, "cursor", None)
        conn = getattr(context, "connection", None)
        key = (id(conn), id(cursor)) if conn and cursor else None
        start = _sa_timings.pop(key, None) if key else None
        duration_ms = (time.perf_counter() - start) * 1000 if start else 0.0
        db_system = _get_sa_db_system(context.engine)
        db_name = context.engine.url.database
        statement = str(getattr(context, "statement", "")) if hasattr(context, "statement") else ""
        operation = _classify_sql(statement)
        host = context.engine.url.host
        port = context.engine.url.port
        error_msg = str(context.original_exception) if hasattr(context, "original_exception") else "Unknown error"

        current_span = otel_trace.get_current_span()
        ident = _db_identifier(db_system, host, port, db_name)
        completed_sd = _build_db_span_data(
            current_span, db_system, db_name, operation, statement,
            host, port, "completed", duration_ms=duration_ms, error=error_msg,
        )
        _evaluate_db_sync(ident, completed_sd, is_completed=True)

    try:
        event.listen(engine, "before_cursor_execute", _before_execute)
        event.listen(engine, "after_cursor_execute", _after_execute)
        event.listen(engine, "handle_error", _on_error)
        _sqlalchemy_listeners.append((engine, "before_cursor_execute", _before_execute))
        _sqlalchemy_listeners.append((engine, "after_cursor_execute", _after_execute))
        _sqlalchemy_listeners.append((engine, "handle_error", _on_error))
        logger.info("DB governance hooks installed: sqlalchemy")
    except (AttributeError, Exception) as e:
        logger.debug(f"Could not register SQLAlchemy governance events: {e}")


def uninstrument_all() -> None:
    """Remove all DB governance hooks (best-effort cleanup)."""
    _uninstall_cursor_tracer_hooks()
    _uninstall_asyncpg_hooks()

    # SQLAlchemy event listeners
    for engine, event_name, listener_fn in _sqlalchemy_listeners:
        try:
            from sqlalchemy import event
            event.remove(engine, event_name, listener_fn)
        except Exception:
            pass
    _sqlalchemy_listeners.clear()

    # Clear tracking
    _installed_patches.clear()
    _pymongo_pending_commands.clear()
    _redis_span_meta.clear()
    _sa_timings.clear()
    logger.info("DB governance hooks uninstrumented")
