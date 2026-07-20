# OpenBox SDK for Temporal Workflows

OpenBox SDK provides **governance and observability** for Temporal workflows by capturing workflow/activity lifecycle events, HTTP telemetry, database queries, and file operations, then sending them to OpenBox Core for policy evaluation.

**Key Features:**
- 7 event types (WorkflowStarted, WorkflowCompleted, WorkflowFailed, SignalReceived, ActivityStarted, ActivityCompleted, Handoff)
- Multi-agent sessions — propagate a shared `multi_agent_session_id` across workflow + activity events
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
- Python 3.11+
- Temporal SDK 1.23+ (1.8+ for factory-only usage)
- OpenTelemetry API/SDK 1.38.0+

---

## Plugin Integration (Recommended)

Use `OpenBoxPlugin` for drop-in integration with Temporal Workers:

```python
import os
from temporalio.worker import Worker
from openbox.plugin import OpenBoxPlugin

worker = Worker(
    client,
    task_queue="my-task-queue",
    workflows=[MyWorkflow],
    activities=[my_activity],
    plugins=[
        OpenBoxPlugin(
            openbox_url=os.getenv("OPENBOX_URL"),
            openbox_api_key=os.getenv("OPENBOX_API_KEY"),
        )
    ],
)

await worker.run()
```

The plugin automatically configures governance interceptors, OTel instrumentation,
sandbox passthrough, W3C trace propagation through Temporal headers (via
`temporalio.contrib.opentelemetry.TracingInterceptor`), and the
`send_governance_event` activity.

**Credentials never leave the plugin.** `openbox_api_key` is captured on the
governance activity instance itself — it does **not** flow through activity
inputs, so it is never written to workflow history. To opt out of trace
propagation (e.g., if you already wire `OpenTelemetryPlugin`), pass
`enable_trace_propagation=False`.

### Composing with Other Plugins

```python
from temporalio.contrib.opentelemetry import OpenTelemetryPlugin

worker = Worker(
    client,
    task_queue="my-task-queue",
    workflows=[MyWorkflow],
    activities=[my_activity],
    plugins=[
        OpenTelemetryPlugin(),
        OpenBoxPlugin(openbox_url=..., openbox_api_key=...),
    ],
)
```

> **Requires** `temporalio >= 1.23.0`. For older versions, use `create_openbox_worker()` below.

---

## Quick Start (Factory)

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
4. Creates governance interceptors (incl. W3C trace propagation)
5. Builds the `GovernanceActivities` instance with credentials captured on
   `self` and registers its `send_governance_event` method — the API key is
   never passed through activity inputs / workflow history
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

    # File I/O instrumentation
    instrument_file_io=False,  # disabled by default

    # Header-based W3C trace propagation (client → workflow → activities).
    # Default True. Set False if you already wire OpenTelemetryPlugin or a
    # custom propagator.
    enable_trace_propagation=True,

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
| Handoff | Agent hands off to another agent | from_agent_did, multi_agent_session_id |

When a workflow runs as part of a multi-agent session (see below), every event
above also carries `multi_agent_session_id`. The field is omitted when no
session id is supplied.

---

## Multi-Agent Sessions

Group the work of several agents under one shared `multi_agent_session_id`. The
SDK only **propagates** the id you supply — it never invents one, and it owns no
routing, session minting, or agent registry (those stay in your application).

**Supply the id** via the Temporal workflow memo at start time:

```python
await client.start_workflow(
    MyWorkflow.run,
    arg,
    id="order-123",
    task_queue="my-queue",
    memo={"openbox_multi_agent_session_id": session_id},
)
```

The SDK then tags every governance event (workflow, activity, and hook events)
with that id, propagating it from the workflow to its activities automatically.

**Emit an explicit handoff** from inside workflow code:

```python
from openbox import emit_handoff

await emit_handoff(
    multi_agent_session_id=session_id,
    from_agent_did="did:aip:...",   # the sending agent
)
```

The receiving agent is derived server-side from the signed identity and is never
sent. `emit_handoff` validates its arguments locally before any network call.

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

Configure error policy via the `governance_policy` parameter (`"fail_open"` / `"fail_closed"`):

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

**SQLAlchemy Note:** Query-level governance works on pre-existing engines automatically — even engines created before `create_openbox_worker()` runs (e.g., at module import time) — because the base runtime governs SQLAlchemy via a global `Engine` event listener. No engine handle needs to be passed in. The `db_libraries` and `sqlalchemy_engine` parameters are still accepted for backward compatibility but are no longer required and have no effect: the base runtime installs every available database instrumentor best-effort.

```python
worker = create_openbox_worker(
    ...,
    instrument_databases=True,  # default; installs all available DB instrumentors
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

Functions decorated with `@traced` are automatically governed when a worker (or plugin) has installed the base runtime. `@traced` wraps the base SDK's `governed()` decorator:

| Stage | Trigger | Data Available |
|-------|---------|----------------|
| `started` | Before function executes | Function name, module, arguments (if `capture_args=True`) |
| `completed` | After function returns/raises | All of above + result (if `capture_result=True`) or error info |

**How it works (Function Tracing):**

1. `@traced` delegates to the base SDK's `governed()` decorator
2. → the base runtime sends a `started` FUNCTION_CALL evaluation with function name and module
3. If verdict is BLOCK/HALT → the call is blocked and the function never executes
4. Function executes normally
5. → the base runtime sends a `completed` FUNCTION_CALL evaluation with result or error info
6. If verdict is BLOCK/HALT → the block is surfaced after execution
7. When no base runtime is installed → transparent passthrough, zero governance calls

---

## Architecture

See [System Architecture](./docs/system-architecture.md) for detailed component design.

**High-Level Flow:**

```
Workflow / Activity lifecycle → Temporal SDK interceptors → OpenBox Core API
                                                            ↓
                                                    Returns Verdict
                                                            ↓
                                            (ALLOW, BLOCK, HALT, REQUIRE_APPROVAL)

Hook-Level (per HTTP request / DB query / file op / @traced fn):
Operation → base-SDK instrumentation → OpenBox Core API (started)   → Allow/Block
          → Operation runs           → OpenBox Core API (completed) → Allow/Block
```

The Temporal worker/plugin builds and owns an `openbox_core` runtime that installs
all hook instrumentation. Hook payload building, evaluation, and enforcement are
owned entirely by the base SDK; the Temporal SDK maps the resulting verdicts onto
Temporal-native effects.

**Responsibility split:**

Base SDK (`openbox_core`):
- HTTP / DB / file / function hook instrumentation
- Hook payload shape, evaluation, and enforcement

Temporal SDK (this package):
- `workflow_interceptor.py` / `activity_interceptor.py` — workflow/activity/signal lifecycle governance events (via workflow-safe activities)
- `governance_state.py` — `TemporalGovernanceState`: signal-verdict bridging, HITL pending-approval markers, completed-hook stop/halt propagation
- `core_adapter.py` — builds the base runtime (`create_core_runtime`) and maps base verdicts to Temporal effects (`TemporalFrameworkAdapter`)

---

## Advanced Usage

The supported integration is a single call. Both `create_openbox_worker(...)` and
`OpenBoxPlugin(...)` build and own the base `openbox_core` runtime, install all
hook instrumentation, register the governance interceptors, and wire the
`send_governance_event` activity for you — there is no manual OpenTelemetry or
span-processor setup to perform.

Worker factory:

```python
import os
from openbox import create_openbox_worker

worker = create_openbox_worker(
    client=client,
    task_queue="my-task-queue",
    workflows=[MyWorkflow],
    activities=[my_activity],
    openbox_url=os.getenv("OPENBOX_URL"),
    openbox_api_key=os.getenv("OPENBOX_API_KEY"),
    governance_policy="fail_closed",
    governance_timeout=30.0,
    instrument_databases=True,
    instrument_file_io=True,
)

await worker.run()
```

Plugin (Temporal >= 1.23.0) — attach to a plain `Worker`:

```python
from temporalio.worker import Worker
from openbox.plugin import OpenBoxPlugin

worker = Worker(
    client,
    task_queue="my-task-queue",
    workflows=[MyWorkflow],
    activities=[my_activity],
    plugins=[OpenBoxPlugin(
        openbox_url=os.getenv("OPENBOX_URL"),
        openbox_api_key=os.getenv("OPENBOX_API_KEY"),
        governance_policy="fail_closed",
    )],
)
```

Both entry points accept the same governance, HITL, signing, and instrumentation
options (see [Configuration](./docs/configuration.md)).

---

## Documentation

- **[Project Overview & PDR](./docs/project-overview-pdr.md)** - Requirements, features, constraints
- **[System Architecture](./docs/system-architecture.md)** - Component design, data flows, security
- **[Codebase Summary](./docs/codebase-summary.md)** - Code structure and component details
- **[Code Standards](./docs/code-standards.md)** - Coding conventions and best practices
- **[Project Roadmap](./docs/project-roadmap.md)** - Future enhancements and timeline

---

## Testing

The SDK includes comprehensive test coverage under `tests/`:

```bash
pytest tests/
```

Coverage spans the worker factory and plugin, workflow/activity/signal
interceptors, HITL approval flow, request signing, public-API compatibility, and
parity with the base SDK's hook governance (`test_temporal_hook_parity.py`,
`test_core_conformance_suite.py`). Hook instrumentation internals (HTTP/DB/file
payload shape) are verified in the base SDK's own conformance suite.

---

## License

MIT License - See LICENSE file for details

---

## Support

- **Issues:** GitHub Issues
- **Documentation:** See `./docs/`

---

**Version:** 1.2.0 | **Last Updated:** 2026-04-05
