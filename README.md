# OpenBox SDK for Temporal Workflows

OpenBox SDK provides **governance and observability** for Temporal workflows by capturing workflow/activity lifecycle events, HTTP telemetry, database queries, and file operations, then sending them to OpenBox Core for policy evaluation.

**Key Features:**
- 6 event types (WorkflowStarted, WorkflowCompleted, WorkflowFailed, SignalReceived, ActivityStarted, ActivityCompleted)
- 5-tier verdict system (ALLOW, CONSTRAIN, REQUIRE_APPROVAL, BLOCK, HALT)
- **Hook-level governance** — per-operation evaluation (HTTP requests, file I/O, database queries, function tracing) with started/completed stages
- HTTP/Database/File I/O instrumentation via OpenTelemetry
- Guardrails: Input/output validation and redaction
- Human-in-the-loop approval with expiration handling
- Zero-code setup via `create_openbox_worker()` factory

---

## Installation

```bash
pip install openbox-temporal-sdk-python
```

**Requirements:**
- Python 3.9+
- Temporal SDK 1.8+
- OpenTelemetry API/SDK 1.38.0+

---

## Quick Start

Use the `create_openbox_worker()` factory for simple integration:

```python
import os
from openbox import create_openbox_worker

worker = create_openbox_worker(
    client=client,
    task_queue="my-task-queue",
    workflows=[MyWorkflow],
    activities=[my_activity],
    # OpenBox config
    openbox_url=os.getenv("OPENBOX_URL"),
    openbox_api_key=os.getenv("OPENBOX_API_KEY"),
)

await worker.run()
```

The factory automatically:
1. Validates the API key
2. Creates span processor
3. Sets up OpenTelemetry instrumentation
4. Creates governance interceptors
5. Adds `send_governance_event` activity
6. Returns fully configured Worker

---

## Configuration

### Environment Variables

```bash
OPENBOX_URL=http://localhost:8086
OPENBOX_API_KEY=obx_test_key_1
OPENBOX_GOVERNANCE_TIMEOUT=30.0
OPENBOX_GOVERNANCE_POLICY=fail_open  # or fail_closed
```

### Factory Function Parameters

```python
worker = create_openbox_worker(
    client=client,
    task_queue="my-task-queue",
    workflows=[MyWorkflow],
    activities=[my_activity],

    # OpenBox config
    openbox_url="http://localhost:8086",
    openbox_api_key="obx_test_key_1",
    governance_timeout=30.0,
    governance_policy="fail_open",

    # Event filtering
    send_start_event=True,
    send_activity_start_event=True,
    skip_workflow_types={"InternalWorkflow"},
    skip_activity_types={"send_governance_event"},
    skip_signals={"heartbeat"},

    # Database instrumentation
    instrument_databases=True,
    db_libraries={"psycopg2", "sqlalchemy"},  # None = all available
    sqlalchemy_engine=engine,  # pass pre-existing engine for query capture

    # File I/O instrumentation
    instrument_file_io=False,  # disabled by default

    # Standard Worker options (all supported)
    activity_executor=my_executor,
    max_concurrent_activities=10,
)
```

---

## Governance Verdicts

OpenBox Core returns a verdict indicating what action the SDK should take.

| Verdict | Behavior |
|---------|----------|
| `ALLOW` | Continue execution normally |
| `CONSTRAIN` | Log constraints, continue |
| `REQUIRE_APPROVAL` | Pause, poll for human approval |
| `BLOCK` | Raise error, stop activity |
| `HALT` | Raise error, terminate workflow |

**v1.0 Backward Compatibility:**
- `"continue"` → `ALLOW`
- `"stop"` → `HALT`
- `"require-approval"` → `REQUIRE_APPROVAL`

---

## Event Types

| Event | Trigger | Captured Fields |
|-------|---------|-----------------|
| WorkflowStarted | Workflow begins | workflow_id, run_id, workflow_type, task_queue |
| WorkflowCompleted | Workflow succeeds | workflow_id, run_id, workflow_type |
| WorkflowFailed | Workflow fails | workflow_id, run_id, workflow_type, error |
| SignalReceived | Signal received | workflow_id, signal_name, signal_args |
| ActivityStarted | Activity begins | activity_id, activity_type, activity_input |
| ActivityCompleted | Activity ends | activity_id, activity_type, activity_input, activity_output, spans, status, duration |

---

## Guardrails (Input/Output Redaction)

OpenBox Core can validate and redact sensitive data before/after activity execution:

```python
# Request
{
  "verdict": "allow",
  "guardrails_result": {
    "input_type": "activity_input",
    "redacted_input": {"prompt": "[REDACTED]", "user_id": "123"},
    "validation_passed": true,
    "reasons": []
  }
}

# If validation fails:
{
  "validation_passed": false,
  "reasons": [
    {"type": "pii", "field": "email", "reason": "Contains PII"}
  ]
}
```

---

## Error Handling

Configure error policy via `on_api_error` (constants available as `hook_governance.FAIL_OPEN` / `FAIL_CLOSED`):

| Policy | Behavior |
|--------|----------|
| `fail_open` (default) | If governance API fails, allow workflow to continue |
| `fail_closed` | If governance API fails, terminate workflow |

---

## Supported Instrumentation

### HTTP Libraries
- `httpx` (sync + async) - full body capture
- `requests` - full body capture
- `urllib3` - full body capture
- `urllib` - request body only

### Databases

**Fully supported** — any dbapi-compatible library using OTel's `CursorTracer.traced_execution()`:
- `psycopg2`, `pymysql`, `mysql-connector-python`, and other dbapi-compliant drivers

**Custom hooks (best-effort)** — these libraries use non-dbapi instrumentation paths; governance hooks may not work correctly in all scenarios:
- `asyncpg` — wrapt wrapper on Connection methods (governance runs outside OTel span context)
- `pymongo` — CommandListener monitoring + wrapt Collection wrappers (dedup via thread-local flag; some internal commands like `endSessions` only produce `completed` stage)
- `redis` — native OTel `request_hook`/`response_hook`
- `sqlalchemy` — `before/after_cursor_execute` event listeners

**SQLAlchemy Note:** If your SQLAlchemy engine is created before `create_openbox_worker()` runs (e.g., at module import time), you must pass it via the `sqlalchemy_engine` parameter. Without this, `SQLAlchemyInstrumentor` only patches future `create_engine()` calls and won't capture queries on pre-existing engines.

```python
from db.engine import engine

worker = create_openbox_worker(
    ...,
    db_libraries={"psycopg2", "sqlalchemy"},
    sqlalchemy_engine=engine,
)
```

### File I/O
- `open()`, `read()`, `write()`, `readline()`, `readlines()`
- Skips system paths (`/dev/`, `/proc/`, `/sys/`, `__pycache__`)

---

## Hook-Level Governance

Every HTTP request, file operation, and database query made during an activity is evaluated by OpenBox Core in real-time at two stages:

### HTTP Requests

| Stage | Trigger | Data Available |
|-------|---------|----------------|
| `started` | Before request is sent | Method, URL, request headers, request body |
| `completed` | After response received | All of above + response headers, response body, status code |

### File Operations

Per-operation governance evaluates **every** `read()`/`write()`/`readline()`/`readlines()`/`writelines()` call, not just open/close:

| Operation | Stage | Trigger | Data Available |
|-----------|-------|---------|----------------|
| `open` | `started` | Before file is opened | File path, open mode |
| `read` | `started` | Before read executes | File path, mode |
| `read` | `completed` | After read returns | data (content read), bytes_read |
| `readline` | `started` | Before readline executes | File path, mode |
| `readline` | `completed` | After readline returns | data (line read), bytes_read |
| `readlines` | `started` | Before readlines executes | File path, mode |
| `readlines` | `completed` | After readlines returns | data (lines read), bytes_read, lines_count |
| `write` | `started` | Before write executes | File path, mode |
| `write` | `completed` | After write returns | data (content written), bytes_written |
| `writelines` | `started` | Before writelines executes | File path, mode |
| `writelines` | `completed` | After writelines returns | data (lines written), bytes_written, lines_count |
| `close` | `completed` | After file is closed | bytes_read, bytes_written, operations list |

**How it works (HTTP):**

1. OTel httpx instrumentation fires a **request hook** → SDK sends `started` governance evaluation with request data
2. If verdict is BLOCK/HALT → request is aborted before it leaves the process
3. After response arrives → SDK sends `completed` governance evaluation with full request+response data
4. If verdict is BLOCK/HALT → `GovernanceBlockedError` is raised, activity fails with `GovernanceStop`

**How it works (File I/O):**

1. Activity calls `open()` → SDK sends `started` governance evaluation with file path and mode
2. If verdict is BLOCK/HALT → file is never opened, `GovernanceBlockedError` is raised
3. Each `read()`/`write()`/`readline()`/`readlines()`/`writelines()` call sends `started` (before) and `completed` (after) governance evaluations — enabling content-based policy enforcement
4. After file is closed → SDK sends `completed` governance with lifecycle summary (total bytes, operations list)
5. File governance requires `instrument_file_io=True` (disabled by default)

A simple open-read-close produces **4 governance evaluations**: open(started) → read(started) → read(completed) → close(completed).

### Database Queries

Every database operation is evaluated at `started` (pre-query, can block) and `completed` (post-query, reports outcome):

| Field | Started | Completed |
|-------|---------|-----------|
| `type` | `"db_query"` | `"db_query"` |
| `stage` | `"started"` | `"completed"` |
| `db_system` | postgresql, mysql, mongodb, redis, sqlite | same |
| `db_name` | Database name | same |
| `db_operation` | SQL verb or command (SELECT, INSERT, GET, etc.) | same |
| `db_statement` | Query string or command | same |
| `server_address` | Host | same |
| `server_port` | Port | same |
| `duration_ms` | — | Query duration in ms |
| `error` | — | Error message or None |

**How it works:**

1. Activity executes a DB query (via any supported library)
2. SDK governance hook intercepts **before** the query → sends `started` evaluation
3. If verdict is BLOCK/HALT → query is aborted, `GovernanceBlockedError` raised
4. Query executes normally → SDK sends `completed` evaluation with duration and error status
5. DB governance is automatic when `instrument_databases=True` (default)

**Per-library strategy:**

| Library | Hook Method | Can Block? | Reliability |
|---------|------------|------------|-------------|
| psycopg2, pymysql, mysql-connector-python | `CursorTracer.traced_execution` patch | Yes | Fully supported |
| asyncpg | `wrapt` wrapper on Connection methods | Yes | Best-effort |
| pymongo | CommandListener + `wrapt` Collection wrappers | Yes (wrapt only) | Best-effort |
| redis | Native OTel `request_hook`/`response_hook` | Yes | Best-effort |
| sqlalchemy | `before/after_cursor_execute` events | Yes | Best-effort |

> **Note:** Libraries marked "Fully supported" use OTel's `CursorTracer`, which guarantees governance hooks run inside the OTel span context. Best-effort libraries use custom hooks that may produce inconsistencies (e.g., missing stages for internal commands). Some C extension types (e.g., `psycopg2.extensions.cursor`) cannot be patched with `wrapt` — in those cases governance hooks are silently skipped, but OTel span capture still works normally.

### Function Tracing

Functions decorated with `@traced` are automatically governed when `hook_governance` is configured:

| Stage | Trigger | Data Available |
|-------|---------|----------------|
| `started` | Before function executes | Function name, module, arguments (if `capture_args=True`) |
| `completed` | After function returns/raises | All of above + result (if `capture_result=True`) or error info |

**How it works (Function Tracing):**

1. `@traced` decorator wrapper starts OTel span
2. → SDK sends `started` governance evaluation with function name and module
3. If verdict is BLOCK/HALT → `GovernanceBlockedError` raised, function never executes
4. Function executes normally
5. → SDK sends `completed` governance evaluation with result or error info
6. If verdict is BLOCK/HALT → `GovernanceBlockedError` raised after execution
7. When `hook_governance` is not configured → zero overhead, no governance calls

**Governed span tracking:** When hook-level governance is active, the SDK marks HTTP spans as "governed" so the OTel `on_end` processor skips buffering them — preventing duplicate spans.

---

## Architecture

See [System Architecture](./docs/system-architecture.md) for detailed component design.

**High-Level Flow:**

```
Workflow/Activity → Interceptors → Span Processor → OpenBox Core API
                                                    ↓
                                            Returns Verdict
                                                    ↓
                                    (ALLOW, BLOCK, HALT, etc.)

Hook-Level (per HTTP request):
Activity HTTP Call → OTel Hook → hook_governance → API (started) → Allow/Block
                   → Response  → hook_governance → API (completed) → Allow/Block

Hook-Level (per file operation):
Activity open()       → hook_governance → API (started)   → Allow/Block
Activity read/write() → hook_governance → API (started)   → Allow/Block
                      → hook_governance → API (completed) → Allow/Block
Activity close()      → hook_governance → API (completed) → Allow/Block

Hook-Level (per DB query):
Activity DB Call → db_governance_hooks → hook_governance → API (started) → Allow/Block
                → Query executes     → hook_governance → API (completed)

Hook-Level (per @traced function):
@traced call    → hook_governance → API (started) → Allow/Block
                → Function runs  → hook_governance → API (completed)
```

**Module responsibilities:**
- `otel_setup.py` — OTel instrumentation, hooks, TracedFile wrapper
- `hook_governance.py` — Shared governance evaluator (payload building, API calls, verdict handling)
- `db_governance_hooks.py` — Per-library DB governance wrappers (wrapt, OTel hooks, SQLAlchemy events)
- `span_processor.py` — Span buffering, activity context tracking
- `activity_interceptor.py` — Activity lifecycle governance events

---

## Advanced Usage

For manual control, import individual components:

```python
from openbox import (
    initialize,
    WorkflowSpanProcessor,
    GovernanceInterceptor,
    GovernanceConfig,
)
from openbox.otel_setup import setup_opentelemetry_for_governance
from openbox.activity_interceptor import ActivityGovernanceInterceptor
from openbox.activities import send_governance_event
from openbox.hook_governance import FAIL_OPEN, FAIL_CLOSED  # error policy constants

# 1. Initialize SDK
initialize(api_url="http://localhost:8086", api_key="obx_test_key_1")

# 2. Create span processor
span_processor = WorkflowSpanProcessor(
    ignored_url_prefixes=["http://localhost:8086"]
)

# 3. Setup OTel instrumentation (governance always enabled)
setup_opentelemetry_for_governance(
    span_processor,
    api_url="http://localhost:8086",
    api_key="obx_test_key_1",
    sqlalchemy_engine=engine,  # optional: instrument pre-existing engine
)

# 4. Create governance config
config = GovernanceConfig(
    on_api_error="fail_closed",
    api_timeout=30.0,
)

# 5. Create interceptors
workflow_interceptor = GovernanceInterceptor(
    api_url="http://localhost:8086",
    api_key="obx_test_key_1",
    span_processor=span_processor,
    config=config,
)

activity_interceptor = ActivityGovernanceInterceptor(
    api_url="http://localhost:8086",
    api_key="obx_test_key_1",
    span_processor=span_processor,
    config=config,
)

# 6. Create worker
from temporalio.worker import Worker
worker = Worker(
    client=client,
    task_queue="my-task-queue",
    workflows=[MyWorkflow],
    activities=[my_activity, send_governance_event],
    interceptors=[workflow_interceptor, activity_interceptor],
)
```

---

## Documentation

- **[Project Overview & PDR](./docs/project-overview-pdr.md)** - Requirements, features, constraints
- **[System Architecture](./docs/system-architecture.md)** - Component design, data flows, security
- **[Codebase Summary](./docs/codebase-summary.md)** - Code structure and component details
- **[Code Standards](./docs/code-standards.md)** - Coding conventions and best practices
- **[Project Roadmap](./docs/project-roadmap.md)** - Future enhancements and timeline

---

## Testing

The SDK includes comprehensive test coverage with 13 test files:

```bash
pytest tests/
```

Test files: `test_activities.py`, `test_activity_interceptor.py`, `test_config.py`, `test_db_governance_hooks.py`, `test_file_governance_hooks.py`, `test_otel_hook_pause.py`, `test_otel_hook_pause_db.py`, `test_otel_setup.py`, `test_span_processor.py`, `test_tracing.py`, `test_types.py`, `test_worker.py`, `test_workflow_interceptor.py`

---

## License

MIT License - See LICENSE file for details

---

## Support

- **Issues:** GitHub Issues
- **Documentation:** See `./docs/`

---

**Version:** 1.1.0 | **Last Updated:** 2026-03-09
