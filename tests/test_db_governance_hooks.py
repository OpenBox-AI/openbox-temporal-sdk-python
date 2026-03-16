"""Tests for database hook-level governance.

Verifies that DB operations trigger governance evaluations at 'started' and
'completed' stages. Covers CursorTracer patch (psycopg2 etc.), redis (native
OTel hooks), sqlalchemy (events), pymongo (CommandListener), and cross-cutting
governance policies.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

import openbox.db_governance_hooks as db_gov
import openbox.hook_governance as hook_gov
from openbox.types import GovernanceBlockedError, Verdict, WorkflowSpanBuffer


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def cleanup_db_hooks():
    """Clean up governance state after each test."""
    yield
    db_gov.uninstrument_all()
    hook_gov._api_url = ""
    hook_gov._api_key = ""
    hook_gov._span_processor = None


def _setup_governance(on_api_error: str = "fail_open") -> MagicMock:
    """Configure hook_governance + db_governance with mocked processor."""
    processor = MagicMock()
    processor.get_activity_context_by_trace.return_value = {
        "workflow_id": "wf-db-1",
        "activity_id": "act-db-1",
    }
    buffer = WorkflowSpanBuffer(
        workflow_id="wf-db-1", run_id="run-1",
        workflow_type="DbWorkflow", task_queue="db-queue",
    )
    processor.get_buffer.return_value = buffer

    hook_gov.configure(
        "http://localhost:9090", "test-key", processor,
        api_timeout=5.0, on_api_error=on_api_error,
    )
    db_gov.configure(processor)
    return processor


@contextmanager
def _mock_httpx_client(verdict="allow", reason=None, side_effect=None):
    """Mock persistent httpx client for governance API calls."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# CursorTracer patch tests (psycopg2, mysql, asyncpg, pymysql)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCursorTracerPatch:
    """Tests for CursorTracer monkey-patch governance hooks."""

    def _make_cursor_tracer(self, db_system="postgresql", database="testdb",
                            host="pg-host", port=5432):
        """Create a mock CursorTracer with db_api_integration."""
        from opentelemetry.instrumentation.dbapi import CursorTracer, DatabaseApiIntegration

        integration = MagicMock(spec=DatabaseApiIntegration)
        integration.database_system = db_system
        integration.database = database
        integration.connection_props = {"host": host, "port": port}
        integration.name = f"{db_system}.{database}"
        integration.capture_parameters = False
        integration.enable_commenter = False
        integration.commenter_options = {}
        integration.enable_attribute_commenter = False
        integration.connect_module = MagicMock()
        integration.span_attributes = {}

        # Create a real tracer for span creation
        from opentelemetry import trace
        integration._tracer = trace.get_tracer("test-tracer")

        return CursorTracer(integration)

    def _make_mock_cursor(self):
        """Create a mock DB cursor."""
        cursor = MagicMock()
        cursor.execute = MagicMock(return_value=None)
        return cursor

    def test_install_patches_cursor_tracer(self):
        """install_cursor_tracer_hooks should patch CursorTracer.traced_execution."""
        from opentelemetry.instrumentation.dbapi import CursorTracer

        original = CursorTracer.traced_execution
        try:
            result = db_gov.install_cursor_tracer_hooks()
            assert result is True
            assert CursorTracer.traced_execution is not original
        finally:
            db_gov._uninstall_cursor_tracer_hooks()

    def test_double_install_is_safe(self):
        """Calling install_cursor_tracer_hooks twice should not double-patch."""
        try:
            db_gov.install_cursor_tracer_hooks()
            db_gov.install_cursor_tracer_hooks()  # should be no-op
            assert db_gov._orig_traced_execution is not None
        finally:
            db_gov._uninstall_cursor_tracer_hooks()

    def test_uninstall_restores_original(self):
        """_uninstall_cursor_tracer_hooks should restore originals."""
        from opentelemetry.instrumentation.dbapi import CursorTracer

        original = CursorTracer.traced_execution
        db_gov.install_cursor_tracer_hooks()
        db_gov._uninstall_cursor_tracer_hooks()
        assert CursorTracer.traced_execution is original

    def test_traced_execution_sends_started_and_completed(self):
        """Patched traced_execution should send started + completed governance."""
        _setup_governance()
        db_gov.install_cursor_tracer_hooks()

        tracer = self._make_cursor_tracer()
        cursor = self._make_mock_cursor()

        with _mock_httpx_client() as mock:
            tracer.traced_execution(
                cursor, cursor.execute, "SELECT * FROM users", None
            )

            assert mock.post.call_count == 2
            started_payload = mock.post.call_args_list[0].kwargs["json"]
            completed_payload = mock.post.call_args_list[1].kwargs["json"]
            assert started_payload["hook_trigger"] is True
            assert completed_payload["hook_trigger"] is True
            started = started_payload["spans"][0]
            completed = completed_payload["spans"][0]
            assert started["stage"] == "started"
            assert started["db_system"] == "postgresql"
            assert started["db_operation"] == "SELECT"
            assert started["db_name"] == "testdb"
            assert started["server_address"] == "pg-host"
            assert completed["stage"] == "completed"
            assert completed["duration_ns"] >= 0

    def test_traced_execution_blocks_on_halt(self):
        """HALT verdict on started should raise GovernanceBlockedError."""
        _setup_governance()
        db_gov.install_cursor_tracer_hooks()

        tracer = self._make_cursor_tracer()
        cursor = self._make_mock_cursor()

        with _mock_httpx_client(verdict="halt", reason="Blocked"):
            with pytest.raises(GovernanceBlockedError):
                tracer.traced_execution(
                    cursor, cursor.execute, "DROP TABLE users", None
                )
            # Query should NOT have been executed
            cursor.execute.assert_not_called()

    def test_traced_execution_classifies_operations(self):
        """Patched traced_execution should classify SQL operations."""
        _setup_governance()
        db_gov.install_cursor_tracer_hooks()

        tracer = self._make_cursor_tracer()
        cursor = self._make_mock_cursor()

        test_cases = [
            ("SELECT 1", "SELECT"),
            ("INSERT INTO t VALUES (1)", "INSERT"),
            ("UPDATE t SET x=1", "UPDATE"),
            ("DELETE FROM t", "DELETE"),
            ("CREATE TABLE t (id INT)", "CREATE"),
        ]
        for query, expected_op in test_cases:
            with _mock_httpx_client() as mock:
                tracer.traced_execution(cursor, cursor.execute, query, None)
                payload = mock.post.call_args_list[0].kwargs["json"]
                span = payload["spans"][0]
                assert span["db_operation"] == expected_op, \
                    f"Expected {expected_op} for query: {query}"

    def test_traced_execution_mysql_system(self):
        """Patched traced_execution should work for mysql db_system."""
        _setup_governance()
        db_gov.install_cursor_tracer_hooks()

        tracer = self._make_cursor_tracer(
            db_system="mysql", database="mydb", host="mysql-host", port=3306
        )
        cursor = self._make_mock_cursor()

        with _mock_httpx_client() as mock:
            tracer.traced_execution(
                cursor, cursor.execute, "SELECT 1", None
            )

            payload = mock.post.call_args_list[0].kwargs["json"]
            started = payload["spans"][0]
            assert started["db_system"] == "mysql"
            assert started["server_address"] == "mysql-host"
            assert started["server_port"] == 3306

    def test_span_data_has_stage_at_root(self):
        """Span data sent to governance API should have 'stage' at root level."""
        processor = _setup_governance()
        db_gov.install_cursor_tracer_hooks()

        tracer = self._make_cursor_tracer()
        cursor = self._make_mock_cursor()

        with _mock_httpx_client() as mock_client:
            tracer.traced_execution(
                cursor, cursor.execute, "SELECT * FROM users", None
            )

        # API should be called 2 times: started + completed
        assert mock_client.post.call_count == 2

        # Check started call
        started_call = mock_client.post.call_args_list[0]
        started_payload = started_call.kwargs["json"]
        started_span = started_payload["spans"][0]

        # Check completed call
        completed_call = mock_client.post.call_args_list[1]
        completed_payload = completed_call.kwargs["json"]
        completed_span = completed_payload["spans"][0]

        # stage must be at ROOT level
        assert started_span["stage"] == "started"
        assert completed_span["stage"] == "completed"

        # stage must NOT be in attributes
        assert "stage" not in started_span.get("attributes", {})
        assert "stage" not in completed_span.get("attributes", {})

        # span_id and trace_id must be present
        assert "span_id" in started_span
        assert "trace_id" in started_span

        # DB-specific fields should be at root level
        assert started_span["db_system"] == "postgresql"
        assert started_span["db_operation"] == "SELECT"

    def test_traced_execution_captures_query_error(self):
        """Patched traced_execution should send completed with error on query failure."""
        _setup_governance()
        db_gov.install_cursor_tracer_hooks()

        tracer = self._make_cursor_tracer()
        cursor = self._make_mock_cursor()

        def failing_execute(*args, **kwargs):
            raise RuntimeError("connection lost")

        with _mock_httpx_client() as mock:
            with pytest.raises(RuntimeError, match="connection lost"):
                tracer.traced_execution(
                    cursor, failing_execute, "SELECT 1", None
                )

            assert mock.post.call_count == 2
            completed_payload = mock.post.call_args_list[1].kwargs["json"]
            completed = completed_payload["spans"][0]
            assert completed["stage"] == "completed"
            assert completed["error"] == "connection lost"


# ═══════════════════════════════════════════════════════════════════════════════
# Redis tests (native OTel hooks — call hook functions directly)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRedisHooks:
    """Tests for redis governance hooks (request_hook / response_hook)."""

    def _make_redis_instance(self):
        """Create a mock Redis instance with connection_pool.connection_kwargs."""
        instance = MagicMock()
        instance.connection_pool.connection_kwargs = {
            "host": "redis-host", "port": 6379, "db": 2,
        }
        return instance

    def test_request_hook_sends_started(self):
        """Redis request_hook should send 'started' governance."""
        _setup_governance()
        req_hook, _ = db_gov.setup_redis_hooks()
        instance = self._make_redis_instance()
        span = MagicMock()

        with _mock_httpx_client() as mock:
            req_hook(span, instance, ("GET", "mykey"), {})

            assert mock.post.call_count == 1
            payload = mock.post.call_args_list[0].kwargs["json"]
            assert payload["hook_trigger"] is True
            span_data = payload["spans"][0]
            assert span_data["hook_type"] == "db_query"
            assert span_data["stage"] == "started"
            assert span_data["db_system"] == "redis"
            assert span_data["db_name"] == "2"
            assert span_data["db_operation"] == "GET"
            assert span_data["db_statement"] == "GET mykey"
            assert span_data["server_address"] == "redis-host"
            assert span_data["server_port"] == 6379

    def test_request_hook_blocks_on_halt(self):
        """Redis request_hook with HALT verdict should raise GovernanceBlockedError."""
        _setup_governance()
        req_hook, _ = db_gov.setup_redis_hooks()
        instance = self._make_redis_instance()
        span = MagicMock()

        with _mock_httpx_client(verdict="halt", reason="Blocked by policy"):
            with pytest.raises(GovernanceBlockedError) as exc_info:
                req_hook(span, instance, ("DEL", "sensitive_key"), {})
            assert exc_info.value.verdict == Verdict.HALT

    def test_response_hook_sends_completed(self):
        """Redis response_hook should send 'completed' governance."""
        _setup_governance()
        req_hook, resp_hook = db_gov.setup_redis_hooks()
        instance = self._make_redis_instance()
        span = MagicMock()

        with _mock_httpx_client() as mock:
            # Call request hook first to record start time
            req_hook(span, instance, ("SET", "key", "val"), {})
            resp_hook(span, instance, "OK")

            # 2 calls: started + completed
            assert mock.post.call_count == 2
            completed_payload = mock.post.call_args_list[1].kwargs["json"]
            completed = completed_payload["spans"][0]
            assert completed["stage"] == "completed"
            assert completed["db_system"] == "redis"
            assert completed["duration_ns"] >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# SQLAlchemy tests (SQLite in-memory — real DB)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSQLAlchemyHooks:
    """Tests for SQLAlchemy governance hooks via before/after_cursor_execute events."""

    def _make_engine(self):
        from sqlalchemy import create_engine
        return create_engine("sqlite:///:memory:")

    def test_select_sends_started_and_completed(self):
        """SQLAlchemy SELECT should produce started + completed governance calls."""
        _setup_governance()
        engine = self._make_engine()
        db_gov.setup_sqlalchemy_hooks(engine)

        with _mock_httpx_client() as mock:
            with engine.connect() as conn:
                conn.execute(
                    __import__("sqlalchemy").text("SELECT 1")
                )

            assert mock.post.call_count == 2
            started_payload = mock.post.call_args_list[0].kwargs["json"]
            completed_payload = mock.post.call_args_list[1].kwargs["json"]
            started = started_payload["spans"][0]
            completed = completed_payload["spans"][0]
            assert started["stage"] == "started"
            assert started["db_operation"] == "SELECT"
            assert started["db_system"] == "sqlite"
            assert completed["stage"] == "completed"
            assert completed["duration_ns"] >= 0

    def test_block_prevents_query(self):
        """BLOCK verdict on started should raise GovernanceBlockedError."""
        _setup_governance()
        engine = self._make_engine()
        db_gov.setup_sqlalchemy_hooks(engine)

        with _mock_httpx_client(verdict="block", reason="Query blocked"):
            with pytest.raises(GovernanceBlockedError):
                with engine.connect() as conn:
                    conn.execute(
                        __import__("sqlalchemy").text("CREATE TABLE forbidden (id INT)")
                    )

    def test_insert_sends_correct_operation(self):
        """INSERT query should report db_operation=INSERT."""
        _setup_governance()
        engine = self._make_engine()
        db_gov.setup_sqlalchemy_hooks(engine)

        with _mock_httpx_client() as mock:
            with engine.connect() as conn:
                conn.execute(
                    __import__("sqlalchemy").text("CREATE TABLE t (id INT)")
                )
                conn.execute(
                    __import__("sqlalchemy").text("INSERT INTO t VALUES (1)")
                )

            insert_calls = [
                c for c in mock.post.call_args_list
                if c.kwargs["json"]["spans"][0]["db_operation"] == "INSERT"
            ]
            assert len(insert_calls) >= 1
            assert insert_calls[0].kwargs["json"]["spans"][0]["stage"] == "started"


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helper tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassifySql:
    """Tests for _classify_sql helper."""

    def test_select(self):
        assert db_gov._classify_sql("SELECT * FROM users") == "SELECT"

    def test_insert(self):
        assert db_gov._classify_sql("INSERT INTO users VALUES (1)") == "INSERT"

    def test_update(self):
        assert db_gov._classify_sql("UPDATE users SET name='x'") == "UPDATE"

    def test_delete(self):
        assert db_gov._classify_sql("DELETE FROM users") == "DELETE"

    def test_unknown(self):
        assert db_gov._classify_sql("MERGE INTO foo") == "UNKNOWN"

    def test_empty(self):
        assert db_gov._classify_sql("") == "UNKNOWN"

    def test_none(self):
        assert db_gov._classify_sql(None) == "UNKNOWN"

    def test_case_insensitive(self):
        assert db_gov._classify_sql("select * from t") == "SELECT"


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-cutting governance policy tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGovernanceDisabled:
    """Tests for governance no-op when not configured or no activity context."""

    def test_no_governance_when_not_configured(self):
        """No API calls when hook_governance is not configured."""
        # Don't call _setup_governance() — leave unconfigured
        hook_gov._api_url = ""
        hook_gov._span_processor = None

        req_hook, resp_hook = db_gov.setup_redis_hooks()
        instance = MagicMock()
        instance.connection_pool.connection_kwargs = {"host": "h", "port": 6379, "db": 0}
        span = MagicMock()

        with patch("openbox.hook_governance._get_sync_client") as mock_get:
            req_hook(span, instance, ("GET", "k"), {})
            resp_hook(span, instance, "OK")
            mock_get.assert_not_called()

    def test_no_governance_outside_activity(self):
        """No API calls when no activity context found."""
        processor = MagicMock()
        processor.get_activity_context_by_trace.return_value = None  # No activity
        hook_gov.configure("http://localhost:9090", "test-key", processor)
        db_gov.configure(processor)

        req_hook, resp_hook = db_gov.setup_redis_hooks()
        instance = MagicMock()
        instance.connection_pool.connection_kwargs = {"host": "h", "port": 6379, "db": 0}
        span = MagicMock()

        with patch("openbox.hook_governance._get_sync_client") as mock_get:
            req_hook(span, instance, ("GET", "k"), {})
            mock_get.assert_not_called()


class TestGovernanceFailPolicy:
    """Tests for fail_open vs fail_closed on governance API errors."""

    def test_fail_open_allows_on_api_error(self):
        """With fail_open, DB operation should proceed if governance API fails."""
        _setup_governance(on_api_error="fail_open")
        req_hook, _ = db_gov.setup_redis_hooks()
        instance = MagicMock()
        instance.connection_pool.connection_kwargs = {"host": "h", "port": 6379, "db": 0}
        span = MagicMock()

        with _mock_httpx_client(side_effect=ConnectionError("API unavailable")):
            # Should not raise — fail_open allows through
            req_hook(span, instance, ("GET", "key"), {})

    def test_fail_closed_blocks_on_api_error(self):
        """With fail_closed, DB operation should be blocked if governance API fails."""
        _setup_governance(on_api_error="fail_closed")
        req_hook, _ = db_gov.setup_redis_hooks()
        instance = MagicMock()
        instance.connection_pool.connection_kwargs = {"host": "h", "port": 6379, "db": 0}
        span = MagicMock()

        with _mock_httpx_client(side_effect=ConnectionError("API unavailable")):
            with pytest.raises(GovernanceBlockedError):
                req_hook(span, instance, ("GET", "key"), {})


# ═══════════════════════════════════════════════════════════════════════════════
# hook_trigger schema validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestHookTriggerSchema:
    """Verify hook_trigger contains all required fields."""

    def test_started_trigger_has_required_fields(self):
        """Span data should have hook_type, stage, db_system, db_operation, db_statement."""
        _setup_governance()
        req_hook, _ = db_gov.setup_redis_hooks()
        instance = MagicMock()
        instance.connection_pool.connection_kwargs = {"host": "h", "port": 6379, "db": 0}
        span = MagicMock()

        with _mock_httpx_client() as mock:
            req_hook(span, instance, ("HSET", "myhash", "field", "value"), {})

            payload = mock.post.call_args_list[0].kwargs["json"]
            assert payload["hook_trigger"] is True
            span_data = payload["spans"][0]
            required = {"hook_type", "stage", "db_system", "db_name", "db_operation",
                        "db_statement", "server_address", "server_port"}
            assert required.issubset(span_data.keys())
            assert span_data["hook_type"] == "db_query"
            assert span_data["stage"] == "started"

    def test_completed_trigger_has_duration_and_error(self):
        """Completed span data must include duration_ns and error fields."""
        _setup_governance()
        req_hook, resp_hook = db_gov.setup_redis_hooks()
        instance = MagicMock()
        instance.connection_pool.connection_kwargs = {"host": "h", "port": 6379, "db": 0}
        span = MagicMock()

        with _mock_httpx_client() as mock:
            req_hook(span, instance, ("GET", "k"), {})
            resp_hook(span, instance, "v")

            payload = mock.post.call_args_list[1].kwargs["json"]
            span_data = payload["spans"][0]
            assert "duration_ns" in span_data
            assert "error" in span_data
            assert span_data["duration_ns"] >= 0
            assert span_data["error"] is None


# ═══════════════════════════════════════════════════════════════════════════════
# pymongo CommandListener tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPymongoCommandListener:
    """Tests for pymongo governance via CommandListener."""

    _next_request_id = 1

    def _make_started_event(self, cmd_name="find", db_name="testdb",
                            command=None, connection_id=None, request_id=None):
        """Create a mock pymongo CommandStartedEvent."""
        event = MagicMock()
        event.command_name = cmd_name
        event.database_name = db_name
        event.command = command or {cmd_name: "collection"}
        event.connection_id = connection_id or ("mongo-host", 27017)
        if request_id is None:
            request_id = self._next_request_id
            self._next_request_id += 1
        event.request_id = request_id
        return event

    def _make_succeeded_event(self, cmd_name="find", db_name="testdb",
                              duration_micros=1500, connection_id=None,
                              request_id=None):
        """Create a mock pymongo CommandSucceededEvent."""
        event = MagicMock()
        event.command_name = cmd_name
        event.database_name = db_name
        event.duration_micros = duration_micros
        event.connection_id = connection_id or ("mongo-host", 27017)
        if request_id is None:
            request_id = self._next_request_id
            self._next_request_id += 1
        event.request_id = request_id
        return event

    def _make_failed_event(self, cmd_name="find", db_name="testdb",
                           duration_micros=500, failure="connection reset",
                           connection_id=None, request_id=None):
        """Create a mock pymongo CommandFailedEvent."""
        event = MagicMock()
        event.command_name = cmd_name
        event.database_name = db_name
        event.duration_micros = duration_micros
        event.failure = failure
        event.connection_id = connection_id or ("mongo-host", 27017)
        if request_id is None:
            request_id = self._next_request_id
            self._next_request_id += 1
        event.request_id = request_id
        return event

    def test_started_sends_governance(self):
        """CommandListener.started should send 'started' governance evaluation."""
        _setup_governance()

        # Create listener instance directly (bypass pymongo.monitoring.register)
        with patch("pymongo.monitoring.register"):
            db_gov.setup_pymongo_hooks()

        # Get the listener from module global
        listener = db_gov._pymongo_listener

        with _mock_httpx_client() as mock:
            listener.started(self._make_started_event(
                cmd_name="insert", db_name="mydb"))

            assert mock.post.call_count == 1
            payload = mock.post.call_args_list[0].kwargs["json"]
            assert payload["hook_trigger"] is True
            span_data = payload["spans"][0]
            assert span_data["stage"] == "started"
            assert span_data["db_system"] == "mongodb"
            assert span_data["db_name"] == "mydb"
            assert span_data["db_operation"] == "insert"
            assert span_data["server_address"] == "mongo-host"
            assert span_data["server_port"] == 27017

    def test_succeeded_sends_completed_with_consistent_statement(self):
        """CommandListener.succeeded should reuse db_statement from started event."""
        _setup_governance()

        with patch("pymongo.monitoring.register"):
            db_gov.setup_pymongo_hooks()

        listener = db_gov._pymongo_listener
        req_id = 42

        with _mock_httpx_client() as mock:
            # Fire started first to store command string
            listener.started(self._make_started_event(
                cmd_name="find", command={"find": "users", "filter": {"age": 30}},
                request_id=req_id))
            listener.succeeded(self._make_succeeded_event(
                cmd_name="find", duration_micros=3000, request_id=req_id))

            assert mock.post.call_count == 2
            started_payload = mock.post.call_args_list[0].kwargs["json"]
            completed_payload = mock.post.call_args_list[1].kwargs["json"]
            # db_statement should be consistent between started and completed
            assert started_payload["spans"][0]["db_statement"] == \
                completed_payload["spans"][0]["db_statement"]
            assert completed_payload["spans"][0]["stage"] == "completed"
            assert completed_payload["spans"][0]["db_system"] == "mongodb"
            assert completed_payload["spans"][0]["duration_ns"] == 3_000_000  # 3ms in nanoseconds
            assert completed_payload["spans"][0]["error"] is None

    def test_failed_sends_completed_with_error_and_consistent_statement(self):
        """CommandListener.failed should send 'completed' with error and consistent db_statement."""
        _setup_governance()

        with patch("pymongo.monitoring.register"):
            db_gov.setup_pymongo_hooks()

        listener = db_gov._pymongo_listener
        req_id = 99

        with _mock_httpx_client() as mock:
            listener.started(self._make_started_event(
                cmd_name="update", command={"update": "users"},
                request_id=req_id))
            listener.failed(self._make_failed_event(
                cmd_name="update", failure="connection timeout",
                request_id=req_id))

            assert mock.post.call_count == 2
            started_payload = mock.post.call_args_list[0].kwargs["json"]
            completed_payload = mock.post.call_args_list[1].kwargs["json"]
            assert started_payload["spans"][0]["db_statement"] == \
                completed_payload["spans"][0]["db_statement"]
            assert completed_payload["spans"][0]["stage"] == "completed"
            assert completed_payload["spans"][0]["error"] == "connection timeout"

    def test_listener_skips_when_wrapt_active(self):
        """CommandListener should skip when wrapt wrapper is active (dedup)."""
        _setup_governance()

        with patch("pymongo.monitoring.register"):
            db_gov.setup_pymongo_hooks()

        listener = db_gov._pymongo_listener

        # Simulate wrapt wrapper being active (depth > 0)
        db_gov._pymongo_wrapt_depth.value = 1
        try:
            with _mock_httpx_client() as mock:
                listener.started(self._make_started_event(cmd_name="find"))
                listener.succeeded(self._make_succeeded_event(cmd_name="find"))
                # No governance calls — wrapt is handling it
                assert mock.post.call_count == 0
        finally:
            db_gov._pymongo_wrapt_depth.value = 0

    def test_listener_fires_when_wrapt_inactive(self):
        """CommandListener should fire normally when wrapt is not active."""
        _setup_governance()

        with patch("pymongo.monitoring.register"):
            db_gov.setup_pymongo_hooks()

        listener = db_gov._pymongo_listener
        req_id = 200

        with _mock_httpx_client() as mock:
            listener.started(self._make_started_event(
                cmd_name="endSessions", request_id=req_id))
            listener.succeeded(self._make_succeeded_event(
                cmd_name="endSessions", request_id=req_id))
            # Both should fire — wrapt doesn't cover endSessions
            assert mock.post.call_count == 2

    def test_address_extraction_fallback(self):
        """Should handle missing connection_id gracefully."""
        _setup_governance()

        with patch("pymongo.monitoring.register"):
            db_gov.setup_pymongo_hooks()

        listener = db_gov._pymongo_listener
        event = self._make_started_event(connection_id=None)
        event.connection_id = None

        with _mock_httpx_client() as mock:
            listener.started(event)

            payload = mock.post.call_args_list[0].kwargs["json"]
            span_data = payload["spans"][0]
            assert span_data["server_address"] == "unknown"
            assert span_data["server_port"] == 27017


# ═══════════════════════════════════════════════════════════════════════════════
# Cleanup tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestUninstrument:
    """Tests for uninstrument_all cleanup."""

    def test_uninstrument_clears_sqlalchemy_listeners(self):
        """uninstrument_all should remove SQLAlchemy event listeners."""
        _setup_governance()
        engine = __import__("sqlalchemy").create_engine("sqlite:///:memory:")
        db_gov.setup_sqlalchemy_hooks(engine)
        assert len(db_gov._sqlalchemy_listeners) == 3  # before, after, handle_error

        db_gov.uninstrument_all()
        assert len(db_gov._sqlalchemy_listeners) == 0

    def test_uninstrument_clears_patch_list(self):
        """uninstrument_all should clear installed patches tracking list."""
        db_gov._installed_patches.append(("test.module", "func"))
        db_gov.uninstrument_all()
        assert len(db_gov._installed_patches) == 0

    def test_uninstrument_restores_cursor_tracer(self):
        """uninstrument_all should restore original CursorTracer methods."""
        from opentelemetry.instrumentation.dbapi import CursorTracer

        original = CursorTracer.traced_execution
        db_gov.install_cursor_tracer_hooks()
        assert CursorTracer.traced_execution is not original

        db_gov.uninstrument_all()
        assert CursorTracer.traced_execution is original
