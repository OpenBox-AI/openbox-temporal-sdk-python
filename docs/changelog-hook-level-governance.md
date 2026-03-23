# v1.1.0 — Hook-Level Governance

Real-time, per-operation governance for every HTTP request, database query, file operation, and traced function call during activity execution.

## What's New

### Hook-Level Governance

Previously, governance only evaluated at activity boundaries (ActivityStarted/Completed). Now, every individual operation inside an activity is evaluated in real-time at two stages:

- **`started`** — before the operation executes. Can block the operation before it runs.
- **`completed`** — after the operation finishes. Informational (operation already executed).

```
Activity starts → ActivityStarted → API → verdict
  HTTP call  → started → allow/block → completed → (report)
  DB query   → started → allow/block → completed → (report)
  File I/O   → started → allow/block → completed → (report)
  @traced fn → started → allow/block → completed → (report)
Activity ends → ActivityCompleted → API → verdict
```

### Supported Operation Types

**HTTP Requests** — httpx (sync + async), requests, urllib3, urllib
- Started: method, URL, request headers/body
- Completed: + response headers/body, status code, duration

**Database Queries** — psycopg2, pymysql, mysql-connector, asyncpg, pymongo, redis, sqlalchemy
- Started: db_system, db_name, db_operation, db_statement, server address/port
- Completed: + duration, rowcount, error

**File Operations** — open, read, write, readline, readlines, writelines, close
- Started: file path, mode, operation type
- Completed: + data content, bytes read/written, lines count
- Opt-in via `instrument_file_io=True` (disabled by default)

**Function Tracing** — `@traced` decorator
- Started: function name, module, arguments (if `capture_args=True`)
- Completed: + result (if `capture_result=True`), error, duration
- Zero overhead when governance is not configured

### Payload Shape

Each hook evaluation sends a single span with all data at root level. `hook_trigger` is a boolean flag:

```json
{
  "source": "workflow-telemetry",
  "workflow_id": "...",
  "run_id": "...",
  "activity_id": "...",
  "activity_type": "...",
  "spans": [{
    "hook_type": "http_request",
    "stage": "completed",
    "http_method": "GET",
    "http_url": "https://api.example.com/data",
    "request_body": "...",
    "response_body": "...",
    "http_status_code": 200,
    "duration_ns": 125000000,
    "attributes": {"http.method": "GET", "http.url": "..."},
    "error": null,
    ...base span fields...
  }],
  "span_count": 1,
  "hook_trigger": true
}
```

### Span Data Interfaces

**Base fields (all types):**
`span_id`, `trace_id`, `parent_span_id`, `name`, `kind`, `stage`, `start_time`, `end_time`, `duration_ns`, `attributes` (OTel-original only), `status`, `events`, `hook_type`, `error`

**HTTP** (`hook_type: "http_request"`): `http_method`, `http_url`, `request_body`, `request_headers`, `response_body`, `response_headers`, `http_status_code`

**DB** (`hook_type: "db_query"`): `db_system`, `db_name`, `db_operation`, `db_statement`, `server_address`, `server_port`, `rowcount`

**File** (`hook_type: "file_operation"`): `file_path`, `file_mode`, `file_operation`, `data`, `bytes_read`, `bytes_written`, `lines_count`

**Function** (`hook_type: "function_call"`): `function`, `module`, `args`, `result`

### `@traced` Decorator

```python
from openbox.tracing import traced

@traced
def my_function(arg1, arg2):
    return do_something(arg1, arg2)

@traced(name="custom-name", capture_args=True, capture_result=True)
async def my_async_function(data):
    return await process(data)
```

### HALT Verdict Workflow Termination

HALT verdicts from hook-level governance now correctly terminate the entire workflow via `client.terminate()`, not just the current activity.

### Hook-Level REQUIRE_APPROVAL

REQUIRE_APPROVAL verdicts from hook-level governance now enter the same human-in-the-loop approval polling flow as activity-level approvals.

## Bug Fixes

- **HALT verdict from hooks not terminating workflow** — Now correctly calls `client.terminate()` to end the workflow.
- **REQUIRE_APPROVAL from hooks not entering approval flow** — Now sets `pending_approval` flag and raises retryable `ApplicationError(type="ApprovalPending")`.
- **Stale buffer/verdict from previous workflow run** — Now checks `run_id` and clears stale state.
- **Subsequent hooks still firing after first block** — Now uses abort propagation to short-circuit all subsequent hooks immediately.

## Breaking Changes

None. Fully backward compatible with v1.0.x. The existing `POST /api/v1/governance/evaluate` endpoint is reused. All v1.0 verdict aliases (`continue`, `stop`, `require-approval`, `action` field) remain supported.

## Configuration

No new configuration required. Hook-level governance is automatically enabled when using `create_openbox_worker()`. The existing `governance_timeout` and `governance_policy` settings apply to hook evaluations.

```python
worker = create_openbox_worker(
    client=client,
    task_queue="my-task-queue",
    workflows=[MyWorkflow],
    activities=[my_activity],
    openbox_url=os.getenv("OPENBOX_URL"),
    openbox_api_key=os.getenv("OPENBOX_API_KEY"),
    governance_timeout=30.0,
    governance_policy="fail_open",
    instrument_file_io=True,
)
```

## For Other SDK Implementors

If your SDK (LangChain, Mastra, etc.) has a similar OpenBox governance flow, here are the key changes to replicate:

1. **Two-stage evaluation** — Send `started` (blocking) and `completed` (informational) governance calls for each operation via `POST /api/v1/governance/evaluate`.
2. **Single span per evaluation** — Each hook sends only the current span in `spans[]`. `hook_trigger: true` signals it's a hook evaluation.
3. **All data at span root** — `hook_type` discriminates the operation type. Type-specific fields (method, url, db_system, file_path, etc.) are at span root level, not in `attributes` or `hook_trigger`.
4. **`attributes` = OTel-original only** — Don't inject custom fields into span attributes.
5. **Activity context resolution** — Map OTel `trace_id` → `(workflow_id, activity_id)` so hook payloads include activity context.
6. **Abort propagation** — Once one hook blocks, short-circuit all subsequent hooks for that activity.
7. **HALT async bridge** — Hooks may run in sync context but `terminate()` is async. Use a flag that the activity interceptor checks in its `finally` block.
