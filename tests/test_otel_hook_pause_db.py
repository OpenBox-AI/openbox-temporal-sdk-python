"""
Quick test to verify whether OTel/SQLAlchemy hooks can pause and block DB queries.

Tests:
  1. SQLAlchemy before_cursor_execute event with sleep (pause test)
  2. SQLAlchemy before_cursor_execute raising exception (block test)

Uses SQLite in-memory — no external DB needed.

Run:
    uv run python tests/test_otel_hook_pause_db.py
"""

import time

from sqlalchemy import create_engine, event, text
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor


SLEEP_SECONDS = 5


class QueryBlockedError(Exception):
    pass


def classify_query(sql: str) -> str:
    normalized = sql.strip().upper()
    if normalized.startswith("SELECT"):
        return "read"
    elif normalized.startswith(("INSERT", "UPDATE", "DELETE")):
        return "write"
    elif normalized.startswith(("CREATE", "ALTER", "DROP", "TRUNCATE")):
        return "ddl"
    return "other"


def _check_result(test_name, elapsed):
    if elapsed >= SLEEP_SECONDS:
        print(f"  >>> YES - {test_name} hook CAN pause the query ({elapsed:.2f}s)\n")
    else:
        print(f"  >>> NO - {test_name} hook CANNOT pause the query ({elapsed:.2f}s)\n")


def test_pause():
    """Test that before_cursor_execute with sleep pauses the query."""
    print(f"{'='*60}")
    print(f"[1/2] SQLAlchemy before_cursor_execute — pause with sleep({SLEEP_SECONDS}s)")
    print(f"{'='*60}")

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)

    engine = create_engine("sqlite:///:memory:")
    SQLAlchemyInstrumentor().instrument(engine=engine)

    # Seed data
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE products (id INTEGER, name TEXT)"))
        conn.execute(text("INSERT INTO products VALUES (1, 'widget')"))
        conn.commit()

    # Add the pause hook
    @event.listens_for(engine, "before_cursor_execute")
    def pause_hook(conn, cursor, statement, parameters, context, executemany):
        query_type = classify_query(statement)
        print(f"  [HOOK] before_cursor_execute: [{query_type.upper()}] {statement[:60]}")
        print(f"  [HOOK] sleeping {SLEEP_SECONDS}s...")
        time.sleep(SLEEP_SECONDS)
        print(f"  [HOOK] sleep done")

    start = time.time()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM products"))
        rows = result.fetchall()
    elapsed = time.time() - start

    print(f"  Result: {rows} | Time: {elapsed:.2f}s")
    _check_result("SQLAlchemy pause", elapsed)

    # Cleanup
    event.remove(engine, "before_cursor_execute", pause_hook)
    SQLAlchemyInstrumentor().uninstrument()
    provider.shutdown()


def test_block():
    """Test that raising in before_cursor_execute blocks the query."""
    print(f"{'='*60}")
    print(f"[2/2] SQLAlchemy before_cursor_execute — block by raising exception")
    print(f"{'='*60}")

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)

    engine = create_engine("sqlite:///:memory:")
    SQLAlchemyInstrumentor().instrument(engine=engine)

    # Seed data
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE products (id INTEGER, name TEXT)"))
        conn.execute(text("INSERT INTO products VALUES (1, 'widget')"))
        conn.commit()

    # Add the blocking hook
    @event.listens_for(engine, "before_cursor_execute")
    def block_hook(conn, cursor, statement, parameters, context, executemany):
        query_type = classify_query(statement)
        print(f"  [HOOK] before_cursor_execute: [{query_type.upper()}] {statement[:60]}")

        # Block DDL
        if query_type == "ddl":
            print(f"  [HOOK] BLOCKING — DDL not allowed")
            raise QueryBlockedError(f"DDL blocked: {statement[:80]}")

        # Block writes
        if query_type == "write":
            print(f"  [HOOK] BLOCKING — writes not allowed")
            raise QueryBlockedError(f"Write blocked: {statement[:80]}")

        print(f"  [HOOK] ALLOWED")

    # SELECT — should pass
    print("\n  --- SELECT (expect: allowed) ---")
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM products"))
            print(f"  Result: {result.fetchall()}")
    except QueryBlockedError as e:
        print(f"  BLOCKED: {e}")

    # INSERT — should block
    print("\n  --- INSERT (expect: blocked) ---")
    try:
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO products VALUES (2, 'gadget')"))
            conn.commit()
            print(f"  Result: insert succeeded (NOT blocked)")
    except QueryBlockedError as e:
        print(f"  BLOCKED: {e}")

    # DROP TABLE — should block
    print("\n  --- DROP TABLE (expect: blocked) ---")
    try:
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE products"))
            conn.commit()
            print(f"  Result: drop succeeded (NOT blocked)")
    except QueryBlockedError as e:
        print(f"  BLOCKED: {e}")

    # DELETE without WHERE — should block
    print("\n  --- DELETE (expect: blocked) ---")
    try:
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM products"))
            conn.commit()
            print(f"  Result: delete succeeded (NOT blocked)")
    except QueryBlockedError as e:
        print(f"  BLOCKED: {e}")

    print()

    # Cleanup
    event.remove(engine, "before_cursor_execute", block_hook)
    SQLAlchemyInstrumentor().uninstrument()
    provider.shutdown()


def main():
    print(f"\nTesting DB query hook pause/block (SQLite in-memory)\n")
    test_pause()
    test_block()
    print("Done.")


if __name__ == "__main__":
    main()
