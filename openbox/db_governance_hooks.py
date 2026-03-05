# openbox/db_governance_hooks.py
"""Hook-level governance for database operations.

Intercepts DB queries at 'started' (pre-query) and 'completed' (post-query)
stages, sending governance evaluations to OpenBox Core via hook_governance.

Supported libraries:
- All dbapi-based (psycopg2, asyncpg, mysql, pymysql) via CursorTracer patch
- pymongo (CommandListener monitoring API)
- redis (native OTel request_hook/response_hook)
- sqlalchemy (before/after_cursor_execute events)

Architecture for dbapi libs: After OTel instrumentors wrap psycopg2.connect()
etc., we monkey-patch CursorTracer.traced_execution to inject governance
hooks around the query_method call (which runs inside the OTel span context).
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
# find_one() internally calls find() — both are wrapped. A depth counter
# (not boolean) prevents the inner wrapper's finally from unblocking
# CommandListener prematurely. CommandListener skips when depth > 0.
_pymongo_wrapt_depth = threading.local()

def is_pymongo_wrapt_active() -> bool:
    """Check if pymongo wrapt wrapper is executing (for OTel span filtering)."""
    return getattr(_pymongo_wrapt_depth, 'value', 0) > 0


# pymongo: store command string from started event (keyed by request_id)
# so succeeded/failed can reuse the same db_statement for consistency.
# Capped to prevent unbounded growth if succeeded/failed events are missed.
_pymongo_pending_commands: Dict[int, str] = {}
_PYMONGO_PENDING_MAX = 1000


_span_processor: Optional["WorkflowSpanProcessor"] = None


def configure(span_processor: "WorkflowSpanProcessor") -> None:
    """Store span_processor reference for mark_governed() and span data building.

    Args:
        span_processor: WorkflowSpanProcessor for governed span tracking
    """
    global _span_processor
    _span_processor = span_processor
    logger.info("DB governance hooks configured")


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_sql(query: Any) -> str:
    """Extract SQL verb from a query string (SELECT, INSERT, UPDATE, etc.)."""
    if not query:
        return "UNKNOWN"
    q = str(query).strip().upper()
    for verb in ("SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP",
                 "ALTER", "TRUNCATE", "BEGIN", "COMMIT", "ROLLBACK", "EXPLAIN"):
        if q.startswith(verb):
            return verb
    return "UNKNOWN"


def _generate_span_id() -> str:
    """Generate a random 16-hex-char span ID for pymongo governance spans."""
    import random
    return format(random.getrandbits(64), "016x")


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
    """Build span data dict for a DB operation (matches _extract_span_data format).

    Creates a span data entry with `stage` at root level for OpenBox Core.
    For 'started' stage: end_time=None, duration_ns=None.
    For 'completed' stage: includes duration and result metadata.

    If gov_span_id is provided, uses it as the span_id and sets the current
    span as parent_span_id (used by pymongo to avoid span_id collisions).
    """
    from . import hook_governance as _hook_gov

    current_span_id, trace_id_hex, default_parent = _hook_gov.extract_span_context(span)

    if gov_span_id:
        # pymongo: use generated span_id, current span becomes parent
        span_id_hex = gov_span_id
        parent_span_id = current_span_id
    else:
        # Default: use current span_id, extract parent normally
        span_id_hex = current_span_id
        parent_span_id = default_parent

    raw_attrs = getattr(span, 'attributes', None)
    attrs = dict(raw_attrs) if raw_attrs and isinstance(raw_attrs, dict) else {}
    # Always set DB-specific attributes
    attrs["db.system"] = db_system
    attrs["db.operation"] = db_operation
    attrs["db.statement"] = db_statement
    if db_name:
        attrs["db.name"] = str(db_name)
    if server_address:
        attrs["server.address"] = server_address
    if server_port:
        attrs["server.port"] = int(server_port)
    if error:
        attrs["openbox.governance.error"] = error
    if rowcount is not None and isinstance(rowcount, int) and rowcount >= 0:
        attrs["db.result.rowcount"] = rowcount
    if duration_ms is not None:
        attrs["openbox.governance.duration_ms"] = round(duration_ms, 2)

    span_name = getattr(span, 'name', None)
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
    }


def _evaluate_started(
    db_system: str,
    db_name: Optional[str],
    db_operation: str,
    db_statement: str,
    server_address: Optional[str],
    server_port: Optional[int],
    span_data: Optional[dict] = None,
) -> None:
    """Send 'started' governance evaluation (sync). Raises GovernanceBlockedError to block."""
    from . import hook_governance as _hook_gov
    if not _hook_gov.is_configured():
        return
    span = otel_trace.get_current_span()
    trigger = {
        "type": "db_query",
        "stage": "started",
        "db_system": db_system,
        "db_name": str(db_name) if db_name else None,
        "db_operation": db_operation,
        "db_statement": db_statement,
        "server_address": server_address,
        "server_port": int(server_port) if server_port else None,
    }
    identifier = f"{db_system}://{server_address or 'unknown'}:{server_port or 0}/{db_name or ''}"
    _hook_gov.evaluate_sync(span, hook_trigger=trigger, identifier=identifier, span_data=span_data)


def _evaluate_completed(
    db_system: str,
    db_name: Optional[str],
    db_operation: str,
    db_statement: str,
    server_address: Optional[str],
    server_port: Optional[int],
    duration_ms: float,
    error: Optional[str],
    span_data: Optional[dict] = None,
) -> None:
    """Send 'completed' governance evaluation (sync). Does not block (query already executed)."""
    from . import hook_governance as _hook_gov
    if not _hook_gov.is_configured():
        return
    span = otel_trace.get_current_span()
    trigger = {
        "type": "db_query",
        "stage": "completed",
        "db_system": db_system,
        "db_name": str(db_name) if db_name else None,
        "db_operation": db_operation,
        "db_statement": db_statement,
        "server_address": server_address,
        "server_port": int(server_port) if server_port else None,
        "duration_ms": round(duration_ms, 2),
        "error": error,
    }
    identifier = f"{db_system}://{server_address or 'unknown'}:{server_port or 0}/{db_name or ''}"
    try:
        _hook_gov.evaluate_sync(span, hook_trigger=trigger, identifier=identifier, span_data=span_data)
    except Exception as e:
        # Completed stage should not block — query already executed
        logger.debug(f"DB governance completed evaluation error (non-blocking): {e}")


async def _evaluate_started_async(
    db_system: str,
    db_name: Optional[str],
    db_operation: str,
    db_statement: str,
    server_address: Optional[str],
    server_port: Optional[int],
    span_data: Optional[dict] = None,
) -> None:
    """Async variant of _evaluate_started."""
    from . import hook_governance as _hook_gov
    if not _hook_gov.is_configured():
        return
    span = otel_trace.get_current_span()
    trigger = {
        "type": "db_query",
        "stage": "started",
        "db_system": db_system,
        "db_name": str(db_name) if db_name else None,
        "db_operation": db_operation,
        "db_statement": db_statement,
        "server_address": server_address,
        "server_port": int(server_port) if server_port else None,
    }
    identifier = f"{db_system}://{server_address or 'unknown'}:{server_port or 0}/{db_name or ''}"
    await _hook_gov.evaluate_async(span, hook_trigger=trigger, identifier=identifier, span_data=span_data)


async def _evaluate_completed_async(
    db_system: str,
    db_name: Optional[str],
    db_operation: str,
    db_statement: str,
    server_address: Optional[str],
    server_port: Optional[int],
    duration_ms: float,
    error: Optional[str],
    span_data: Optional[dict] = None,
) -> None:
    """Async variant of _evaluate_completed."""
    from . import hook_governance as _hook_gov
    if not _hook_gov.is_configured():
        return
    span = otel_trace.get_current_span()
    trigger = {
        "type": "db_query",
        "stage": "completed",
        "db_system": db_system,
        "db_name": str(db_name) if db_name else None,
        "db_operation": db_operation,
        "db_statement": db_statement,
        "server_address": server_address,
        "server_port": int(server_port) if server_port else None,
        "duration_ms": round(duration_ms, 2),
        "error": error,
    }
    identifier = f"{db_system}://{server_address or 'unknown'}:{server_port or 0}/{db_name or ''}"
    try:
        await _hook_gov.evaluate_async(span, hook_trigger=trigger, identifier=identifier, span_data=span_data)
    except Exception as e:
        logger.debug(f"DB governance completed evaluation error (non-blocking): {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# CursorTracer patch — intercepts ALL dbapi query execution
# ═══════════════════════════════════════════════════════════════════════════════
#
# OTel dbapi instrumentors (psycopg2, asyncpg, mysql, pymysql) silently
# discard request_hook/response_hook kwargs. Instead, we monkey-patch
# CursorTracer.traced_execution AFTER OTel instruments, injecting governance
# hooks around the query_method call (which runs inside the OTel span context).
# ═══════════════════════════════════════════════════════════════════════════════

# Saved originals for uninstrumentation
_orig_traced_execution: Optional[Callable] = None
_orig_traced_execution_async: Optional[Callable] = None


def install_cursor_tracer_hooks() -> bool:
    """Monkey-patch OTel CursorTracer to inject governance hooks.

    Must be called AFTER OTel dbapi instrumentors are set up.
    Patches traced_execution and traced_execution_async so governance
    evaluations fire inside the OTel span context.

    Returns True if patch was applied, False otherwise.
    """
    global _orig_traced_execution, _orig_traced_execution_async

    try:
        from opentelemetry.instrumentation.dbapi import CursorTracer
    except ImportError:
        logger.debug("OTel dbapi not available for CursorTracer patching")
        return False

    # Guard against double-patching
    if _orig_traced_execution is not None:
        logger.debug("CursorTracer already patched — skipping")
        return True

    _orig_traced_execution = CursorTracer.traced_execution
    _orig_traced_execution_async = CursorTracer.traced_execution_async

    def _gov_traced_execution(self, cursor, query_method, *args, **kwargs):
        """Wrapped traced_execution with governance hooks."""
        db_system = self._db_api_integration.database_system
        db_name = self._db_api_integration.database
        query = args[0] if args else ""
        operation = _classify_sql(query)
        stmt = str(query)[:2000]
        host = self._db_api_integration.connection_props.get("host", "unknown")
        port = self._db_api_integration.connection_props.get("port")

        def _governed_query(*qargs, **qkwargs):
            # Runs inside OTel span context — get_current_span() returns DB span
            current_span = otel_trace.get_current_span()

            # Mark span as governed — on_end() will skip buffering it
            from . import hook_governance as _hg
            _hg.mark_span_governed(current_span)

            # Build & send "started" span data entry
            started_sd = _build_db_span_data(
                current_span, db_system, db_name, operation, stmt, host, port, "started",
            )
            _evaluate_started(db_system, db_name, operation, stmt, host, port, span_data=started_sd)

            start = time.perf_counter()
            try:
                result = query_method(*qargs, **qkwargs)
                duration_ms = (time.perf_counter() - start) * 1000

                # Capture rowcount
                rc = None
                try:
                    rc = getattr(cursor, "rowcount", -1)
                    if rc is None or rc < 0:
                        rc = None
                except Exception:
                    pass

                # Build & send "completed" span data entry
                completed_sd = _build_db_span_data(
                    current_span, db_system, db_name, operation, stmt, host, port,
                    "completed", duration_ms=duration_ms, rowcount=rc,
                )
                _evaluate_completed(
                    db_system, db_name, operation, stmt, host, port,
                    duration_ms, None, span_data=completed_sd,
                )
                return result
            except Exception as e:
                from .types import GovernanceBlockedError
                if isinstance(e, GovernanceBlockedError):
                    raise
                duration_ms = (time.perf_counter() - start) * 1000
                completed_sd = _build_db_span_data(
                    current_span, db_system, db_name, operation, stmt, host, port,
                    "completed", duration_ms=duration_ms, error=str(e),
                )
                _evaluate_completed(
                    db_system, db_name, operation, stmt, host, port,
                    duration_ms, str(e), span_data=completed_sd,
                )
                raise

        return _orig_traced_execution(self, cursor, _governed_query, *args, **kwargs)

    async def _gov_traced_execution_async(self, cursor, query_method, *args, **kwargs):
        """Wrapped traced_execution_async with governance hooks."""
        db_system = self._db_api_integration.database_system
        db_name = self._db_api_integration.database
        query = args[0] if args else ""
        operation = _classify_sql(query)
        stmt = str(query)[:2000]
        host = self._db_api_integration.connection_props.get("host", "unknown")
        port = self._db_api_integration.connection_props.get("port")

        async def _governed_query_async(*qargs, **qkwargs):
            current_span = otel_trace.get_current_span()

            # Mark governed — on_end() skips buffering
            from . import hook_governance as _hg
            _hg.mark_span_governed(current_span)

            started_sd = _build_db_span_data(
                current_span, db_system, db_name, operation, stmt, host, port, "started",
            )
            await _evaluate_started_async(db_system, db_name, operation, stmt, host, port, span_data=started_sd)

            start = time.perf_counter()
            try:
                result = await query_method(*qargs, **qkwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                rc = None
                try:
                    rc = getattr(cursor, "rowcount", -1)
                    if rc is None or rc < 0:
                        rc = None
                except Exception:
                    pass
                completed_sd = _build_db_span_data(
                    current_span, db_system, db_name, operation, stmt, host, port,
                    "completed", duration_ms=duration_ms, rowcount=rc,
                )
                await _evaluate_completed_async(
                    db_system, db_name, operation, stmt, host, port,
                    duration_ms, None, span_data=completed_sd,
                )
                return result
            except Exception as e:
                from .types import GovernanceBlockedError
                if isinstance(e, GovernanceBlockedError):
                    raise
                duration_ms = (time.perf_counter() - start) * 1000
                completed_sd = _build_db_span_data(
                    current_span, db_system, db_name, operation, stmt, host, port,
                    "completed", duration_ms=duration_ms, error=str(e),
                )
                await _evaluate_completed_async(
                    db_system, db_name, operation, stmt, host, port,
                    duration_ms, str(e), span_data=completed_sd,
                )
                raise

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
# asyncpg — wrapt wrapper AFTER OTel (asyncpg doesn't use CursorTracer)
# ═══════════════════════════════════════════════════════════════════════════════

_asyncpg_patched = False


def install_asyncpg_hooks() -> bool:
    """Install governance hooks on asyncpg via wrapt wrapping.

    asyncpg's OTel instrumentor uses its own _do_execute (not CursorTracer),
    so we wrap Connection methods with wrapt AFTER OTel instruments. Our
    wrapper is outermost: governance → OTel → raw asyncpg method.

    Must be called AFTER AsyncPGInstrumentor().instrument().
    """
    global _asyncpg_patched
    if _asyncpg_patched:
        return True

    try:
        import wrapt
        import asyncpg  # noqa: F401 — verify asyncpg is installed
    except ImportError:
        logger.debug("asyncpg or wrapt not available for governance hooks")
        return False

    async def _asyncpg_governance_wrapper(wrapped, instance, args, kwargs):
        """Wrapt wrapper for asyncpg Connection methods with governance."""
        query = args[0] if args else ""
        operation = _classify_sql(query)
        stmt = str(query)[:2000]

        # Extract connection metadata
        params = getattr(instance, "_params", None)
        host = getattr(instance, "_addr", ("unknown",))[0] if hasattr(instance, "_addr") else "unknown"
        port = getattr(instance, "_addr", (None, 5432))[1] if hasattr(instance, "_addr") else 5432
        db_name = getattr(params, "database", None) if params else None

        current_span = otel_trace.get_current_span()
        from . import hook_governance as _hg
        _hg.mark_span_governed(current_span)

        started_sd = _build_db_span_data(
            current_span, "postgresql", db_name, operation, stmt, host, port, "started",
        )
        await _evaluate_started_async("postgresql", db_name, operation, stmt, host, port, span_data=started_sd)
        start = time.perf_counter()
        try:
            result = await wrapped(*args, **kwargs)
            duration_ms = (time.perf_counter() - start) * 1000
            completed_sd = _build_db_span_data(
                current_span, "postgresql", db_name, operation, stmt, host, port,
                "completed", duration_ms=duration_ms,
            )
            await _evaluate_completed_async(
                "postgresql", db_name, operation, stmt, host, port, duration_ms, None, span_data=completed_sd,
            )
            return result
        except Exception as e:
            from .types import GovernanceBlockedError
            if isinstance(e, GovernanceBlockedError):
                raise
            duration_ms = (time.perf_counter() - start) * 1000
            completed_sd = _build_db_span_data(
                current_span, "postgresql", db_name, operation, stmt, host, port,
                "completed", duration_ms=duration_ms, error=str(e),
            )
            await _evaluate_completed_async(
                "postgresql", db_name, operation, stmt, host, port, duration_ms, str(e), span_data=completed_sd,
            )
            raise

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
    # wrapt patches can't be cleanly unwrapped — clear tracking only
    _asyncpg_patched = False


# ═══════════════════════════════════════════════════════════════════════════════
# pymongo (CommandListener — reliable monitoring for all pymongo versions)
# ═══════════════════════════════════════════════════════════════════════════════

# Track pymongo listener reference for cleanup
_pymongo_listener: Any = None


def setup_pymongo_hooks() -> None:
    """Install governance hooks on pymongo via monitoring.CommandListener.

    Uses pymongo's native monitoring API instead of wrapt wrapping, which is
    more reliable across pymongo versions and C extension boundaries.

    Note: CommandListener can monitor but cannot block operations (pymongo
    swallows listener exceptions). For blocking support, we also wrap
    Collection methods with wrapt where possible.

    IMPORTANT: pymongo.monitoring.register() must be called BEFORE creating
    MongoClient instances. Ensure setup_opentelemetry_for_governance() is
    called early in application startup.
    """
    global _pymongo_listener
    try:
        import pymongo.monitoring

        class _GovernanceCommandListener(pymongo.monitoring.CommandListener):
            """Pymongo CommandListener that sends governance evaluations.

            Skips operations already governed by wrapt wrappers (dedup).
            When wrapt is active (depth > 0), marks OTel pymongo spans as
            governed so they don't appear as separate entries in the buffer.
            Stores command string from started event so succeeded/failed
            can reuse the same db_statement for consistency.
            """

            def _mark_otel_span_governed(self):
                """Mark the current OTel pymongo span as governed (suppress from buffer)."""
                try:
                    from . import hook_governance as _hg
                    _hg.mark_span_governed(otel_trace.get_current_span())
                except Exception:
                    pass

            def started(self, event):
                # Skip if wrapt wrapper is already handling this operation
                if getattr(_pymongo_wrapt_depth, 'value', 0) > 0:
                    self._mark_otel_span_governed()
                    return
                try:
                    span = otel_trace.get_current_span()
                    host, port = _extract_pymongo_address(event)
                    cmd_str = str(event.command)[:2000]
                    # Store command string for reuse in succeeded/failed (capped)
                    if len(_pymongo_pending_commands) >= _PYMONGO_PENDING_MAX:
                        _pymongo_pending_commands.clear()
                    _pymongo_pending_commands[event.request_id] = cmd_str
                    started_sd = _build_db_span_data(
                        span, "mongodb", event.database_name, event.command_name,
                        cmd_str, host, port, "started",
                    )
                    _evaluate_started(
                        "mongodb", event.database_name, event.command_name,
                        cmd_str, host, port, span_data=started_sd,
                    )
                except Exception as e:
                    logger.debug(f"pymongo governance started error: {e}")

            def succeeded(self, event):
                # Skip if wrapt wrapper is already handling this operation
                if getattr(_pymongo_wrapt_depth, 'value', 0) > 0:
                    _pymongo_pending_commands.pop(event.request_id, None)
                    return
                try:
                    span = otel_trace.get_current_span()
                    host, port = _extract_pymongo_address(event)
                    duration_ms = event.duration_micros / 1000.0
                    # Reuse command string from started event for consistency
                    cmd_str = _pymongo_pending_commands.pop(event.request_id, event.command_name)
                    completed_sd = _build_db_span_data(
                        span, "mongodb", event.database_name, event.command_name,
                        cmd_str, host, port, "completed", duration_ms=duration_ms,
                    )
                    _evaluate_completed(
                        "mongodb", event.database_name, event.command_name,
                        cmd_str, host, port, duration_ms, None, span_data=completed_sd,
                    )
                except Exception as e:
                    logger.debug(f"pymongo governance completed error: {e}")

            def failed(self, event):
                # Skip if wrapt wrapper is already handling this operation
                if getattr(_pymongo_wrapt_depth, 'value', 0) > 0:
                    _pymongo_pending_commands.pop(event.request_id, None)
                    return
                try:
                    span = otel_trace.get_current_span()
                    host, port = _extract_pymongo_address(event)
                    duration_ms = event.duration_micros / 1000.0
                    err = str(event.failure)
                    # Reuse command string from started event for consistency
                    cmd_str = _pymongo_pending_commands.pop(event.request_id, event.command_name)
                    completed_sd = _build_db_span_data(
                        span, "mongodb", event.database_name, event.command_name,
                        cmd_str, host, port, "completed", duration_ms=duration_ms, error=err,
                    )
                    _evaluate_completed(
                        "mongodb", event.database_name, event.command_name,
                        cmd_str, host, port, duration_ms, err, span_data=completed_sd,
                    )
                except Exception as e:
                    logger.debug(f"pymongo governance failed error: {e}")

        _pymongo_listener = _GovernanceCommandListener()
        pymongo.monitoring.register(_pymongo_listener)
        logger.info("DB governance hooks installed: pymongo (CommandListener)")
    except ImportError:
        logger.debug("pymongo not available for governance hooks")

    # Also try wrapt wrapping for blocking support (best-effort)
    _setup_pymongo_wrapt_hooks()


def _extract_pymongo_address(event) -> Tuple[str, int]:
    """Extract (host, port) from a pymongo monitoring event."""
    try:
        addr = event.connection_id  # (host, port) tuple
        if addr and len(addr) >= 2:
            return str(addr[0]), int(addr[1])
    except (AttributeError, TypeError, IndexError):
        pass
    return "unknown", 27017


def _setup_pymongo_wrapt_hooks() -> None:
    """Best-effort wrapt wrapping of pymongo Collection methods for blocking."""
    try:
        import wrapt
        from .types import GovernanceBlockedError

        def _collection_wrapper(wrapped, instance, args, kwargs):
            # Increment depth counter — tracks nesting (find_one → find internally).
            # Only the outermost call (depth 0→1) fires governance.
            # Inner calls and CommandListener are suppressed when depth > 0.
            depth = getattr(_pymongo_wrapt_depth, 'value', 0)
            _pymongo_wrapt_depth.value = depth + 1

            # Nested call — just pass through, outer wrapper handles governance
            if depth > 0:
                try:
                    return wrapped(*args, **kwargs)
                finally:
                    _pymongo_wrapt_depth.value = getattr(_pymongo_wrapt_depth, 'value', 1) - 1

            # Outermost call — fire governance
            db_name = instance.database.name
            operation = wrapped.__name__
            try:
                address = instance.database.client.address
                host, port = address[0], address[1]
            except (AttributeError, TypeError):
                host, port = "unknown", 27017
            statement = f"{instance.name}.{operation}"

            current_span = otel_trace.get_current_span()

            # Mark activity span as governed
            from . import hook_governance as _hg
            _hg.mark_span_governed(current_span)

            # Generate unique span_id for this pymongo operation (shared by started+completed)
            gov_sid = _generate_span_id()

            started_sd = _build_db_span_data(
                current_span, "mongodb", db_name, operation, statement, host, port,
                "started", gov_span_id=gov_sid,
            )
            _evaluate_started("mongodb", db_name, operation, statement, host, port, span_data=started_sd)
            start = time.perf_counter()
            try:
                result = wrapped(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                completed_sd = _build_db_span_data(
                    current_span, "mongodb", db_name, operation, statement, host, port,
                    "completed", duration_ms=duration_ms, gov_span_id=gov_sid,
                )
                _evaluate_completed("mongodb", db_name, operation, statement, host, port, duration_ms, None, span_data=completed_sd)
                return result
            except GovernanceBlockedError:
                raise
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                completed_sd = _build_db_span_data(
                    current_span, "mongodb", db_name, operation, statement, host, port,
                    "completed", duration_ms=duration_ms, error=str(e), gov_span_id=gov_sid,
                )
                _evaluate_completed("mongodb", db_name, operation, statement, host, port, duration_ms, str(e), span_data=completed_sd)
                raise
            finally:
                _pymongo_wrapt_depth.value = getattr(_pymongo_wrapt_depth, 'value', 1) - 1

        methods = ("find", "find_one", "insert_one", "insert_many",
                   "update_one", "update_many", "delete_one", "delete_many",
                   "aggregate", "count_documents")
        patched = 0
        for method in methods:
            try:
                wrapt.wrap_function_wrapper("pymongo.collection", f"Collection.{method}", _collection_wrapper)
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
# redis (native OTel hooks — returns callables for RedisInstrumentor)
# ═══════════════════════════════════════════════════════════════════════════════

_redis_span_meta: Dict[int, Tuple[float, str, str, str, int, str]] = {}


def setup_redis_hooks() -> Tuple[Callable, Callable]:
    """Return (request_hook, response_hook) for RedisInstrumentor.instrument().

    request_hook fires at 'started' stage (can raise GovernanceBlockedError).
    response_hook fires at 'completed' stage.
    """

    def _request_hook(span, instance, args, kwargs):
        """OTel Redis request hook — 'started' stage."""
        command = str(args[0]) if args else "UNKNOWN"
        statement = " ".join(str(a) for a in args) if args else ""
        try:
            conn_kwargs = instance.connection_pool.connection_kwargs
            host = conn_kwargs.get("host", "localhost")
            port = conn_kwargs.get("port", 6379)
            db_name = str(conn_kwargs.get("db", 0))
        except AttributeError:
            host, port, db_name = "localhost", 6379, "0"

        # Mark governed — we create our own started/completed span entries
        from . import hook_governance as _hg
        _hg.mark_span_governed(span)

        started_sd = _build_db_span_data(span, "redis", db_name, command, statement, host, port, "started")
        _evaluate_started("redis", db_name, command, statement, host, port, span_data=started_sd)
        _redis_span_meta[id(span)] = (time.perf_counter(), command, statement, host, port, db_name)

    def _response_hook(span, instance, response):
        """OTel Redis response hook — 'completed' stage."""
        meta = _redis_span_meta.pop(id(span), None)
        start_time = meta[0] if meta else time.perf_counter()
        command = meta[1] if meta else "UNKNOWN"
        statement = meta[2] if meta else ""
        host = meta[3] if meta and len(meta) > 3 else "localhost"
        port = meta[4] if meta and len(meta) > 4 else 6379
        db_name = meta[5] if meta and len(meta) > 5 else "0"
        duration_ms = (time.perf_counter() - start_time) * 1000

        completed_sd = _build_db_span_data(
            span, "redis", db_name, command, statement, host, port,
            "completed", duration_ms=duration_ms,
        )
        _evaluate_completed("redis", db_name, command, statement, host, port, duration_ms, None, span_data=completed_sd)

    return _request_hook, _response_hook


# ═══════════════════════════════════════════════════════════════════════════════
# sqlalchemy (native SQLAlchemy before/after_cursor_execute events)
# ═══════════════════════════════════════════════════════════════════════════════

# Per-cursor timing for SQLAlchemy (maps (conn_id, cursor_id) → start time)
_sa_timings: Dict[Tuple[int, int], float] = {}


def _get_sa_db_system(engine) -> str:
    """Extract db_system from SQLAlchemy engine dialect name."""
    dialect = getattr(engine, "dialect", None)
    name = getattr(dialect, "name", "") if dialect else ""
    mapping = {"postgresql": "postgresql", "mysql": "mysql", "sqlite": "sqlite",
               "oracle": "oracle", "mssql": "mssql"}
    return mapping.get(name, name or "unknown")


def setup_sqlalchemy_hooks(engine) -> None:
    """Register SQLAlchemy event listeners for governance on the given engine."""
    try:
        from sqlalchemy import event
        from sqlalchemy.engine import Engine as _SAEngine
    except ImportError:
        logger.debug("sqlalchemy not available for governance hooks")
        return

    # Only register on real SQLAlchemy Engine instances (not mocks in tests)
    if not isinstance(engine, _SAEngine):
        logger.debug("Skipping SQLAlchemy governance hooks: not a real Engine instance")
        return

    def _before_execute(conn, cursor, statement, parameters, context, executemany):
        _sa_timings[(id(conn), id(cursor))] = time.perf_counter()
        db_system = _get_sa_db_system(conn.engine)
        db_name = conn.engine.url.database
        operation = _classify_sql(statement)
        host = conn.engine.url.host
        port = conn.engine.url.port

        current_span = otel_trace.get_current_span()
        from . import hook_governance as _hg
        _hg.mark_span_governed(current_span)

        started_sd = _build_db_span_data(
            current_span, db_system, db_name, operation, str(statement), host, port, "started",
        )
        _evaluate_started(db_system, db_name, operation, str(statement), host, port, span_data=started_sd)

    def _after_execute(conn, cursor, statement, parameters, context, executemany):
        start = _sa_timings.pop((id(conn), id(cursor)), None)
        duration_ms = (time.perf_counter() - start) * 1000 if start else 0.0
        db_system = _get_sa_db_system(conn.engine)
        db_name = conn.engine.url.database
        operation = _classify_sql(statement)
        host = conn.engine.url.host
        port = conn.engine.url.port

        current_span = otel_trace.get_current_span()
        completed_sd = _build_db_span_data(
            current_span, db_system, db_name, operation, str(statement), host, port,
            "completed", duration_ms=duration_ms,
        )
        _evaluate_completed(db_system, db_name, operation, str(statement), host, port, duration_ms, None, span_data=completed_sd)

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
        completed_sd = _build_db_span_data(
            current_span, db_system, db_name, operation, statement, host, port,
            "completed", duration_ms=duration_ms, error=error_msg,
        )
        _evaluate_completed(db_system, db_name, operation, statement, host, port, duration_ms, error_msg, span_data=completed_sd)

    try:
        event.listen(engine, "before_cursor_execute", _before_execute)
        event.listen(engine, "after_cursor_execute", _after_execute)
        event.listen(engine, "handle_error", _on_error)
        _sqlalchemy_listeners.append((engine, "before_cursor_execute", _before_execute))
        _sqlalchemy_listeners.append((engine, "after_cursor_execute", _after_execute))
        _sqlalchemy_listeners.append((engine, "handle_error", _on_error))
        logger.info("DB governance hooks installed: sqlalchemy")
    except (AttributeError, Exception) as e:
        # autospec mocks or incomplete Engine objects may fail event registration
        logger.debug(f"Could not register SQLAlchemy governance events: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Cleanup
# ═══════════════════════════════════════════════════════════════════════════════

def uninstrument_all() -> None:
    """Remove all installed DB governance hooks."""
    # Restore original CursorTracer methods
    _uninstall_cursor_tracer_hooks()
    _uninstall_asyncpg_hooks()

    # Remove SQLAlchemy event listeners
    for engine, event_name, listener_fn in _sqlalchemy_listeners:
        try:
            from sqlalchemy import event
            event.remove(engine, event_name, listener_fn)
        except Exception:
            pass
    _sqlalchemy_listeners.clear()

    # wrapt patches can't be cleanly removed — clear the list for bookkeeping
    _installed_patches.clear()

    # Clear timing/tracking dicts
    _sa_timings.clear()
    _pymongo_pending_commands.clear()
    _redis_span_meta.clear()

    logger.info("DB governance hooks removed")
