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
- Completed: + response headers/body, status code

**Database Queries** — psycopg2, pymysql, mysql-connector, asyncpg, pymongo, redis, sqlalchemy
- Started: db_system, db_name, db_operation, db_statement, server address/port
- Completed: + duration_ms, error

**File Operations** — open, read, write, readline, readlines, writelines, close
- Started: file path, mode, operation type
- Completed: + data content, bytes read/written, lines count
- Opt-in via `instrument_file_io=True` (disabled by default)

**Function Tracing** — `@traced` decorator
- Started: function name, module, arguments (if `capture_args=True`)
- Completed: + result (if `capture_result=True`), error
- Zero overhead when governance is not configured

### Hook Trigger Payload

Governance evaluations now include a `hook_trigger` object:

```json
{
  "source": "workflow-telemetry",
  "workflow_id": "...",
  "run_id": "...",
  "activity_id": "...",
  "activity_type": "...",
  "hook_trigger": {
    "type": "http_request | file_operation | db_query | function_call",
    "stage": "started | completed",
    "attribute_key_identifiers": ["http.method", "http.url"]
  }
}
```

`attribute_key_identifiers` are OTel semantic convention keys that uniquely identify a span by its type. The OpenBox server uses these to detect and discard duplicate spans (e.g. from Temporal activity retries or OTel instrumentation overlap).

| Type | Dedup Keys |
|------|-----------|
| HTTP | `http.method`, `http.url` |
| File | `file.path`, `file.operation` |
| DB | `db.system`, `db.operation`, `db.statement` |
| Function | `code.function`, `code.namespace` |

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

- **HALT verdict from hooks not terminating workflow** — HALT was treated the same as BLOCK (only stopped the activity). Now correctly calls `client.terminate()` to end the workflow.
- **REQUIRE_APPROVAL from hooks not entering approval flow** — `GovernanceBlockedError` with REQUIRE_APPROVAL was unhandled. Now sets `pending_approval` flag and raises retryable `ApplicationError(type="ApprovalPending")`.
- **Stale buffer/verdict from previous workflow run** — Buffer and verdict from a previous run could carry over when workflow_id was reused. Now checks `run_id` and clears stale state.
- **Subsequent hooks still firing after first block** — After one hook blocked, remaining hooks for the same activity still executed. Now uses abort propagation to short-circuit all subsequent hooks immediately.

## Breaking Changes

None. Fully backward compatible with v1.0.x. The existing `POST /api/v1/governance/evaluate` endpoint is reused with the additional `hook_trigger` field. All v1.0 verdict aliases (`continue`, `stop`, `require-approval`, `action` field) remain supported.

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
    # These settings now also apply to hook-level governance
    governance_timeout=30.0,
    governance_policy="fail_open",
    # Opt-in for file I/O governance
    instrument_file_io=True,
)
```

## For Other SDK Implementors

If your SDK (LangChain, Mastra, etc.) has a similar OpenBox governance flow, here are the key changes to replicate:

1. **Two-stage evaluation** — Send `started` (blocking) and `completed` (informational) governance calls for each operation, using the same `/api/v1/governance/evaluate` endpoint with a `hook_trigger` field.
2. **Activity context resolution** — Map OTel `trace_id` → `(workflow_id, activity_id)` so hook payloads include activity context.
3. **Abort propagation** — Once one hook blocks, short-circuit all subsequent hooks for that activity.
4. **Governed span dedup** — Mark individually-evaluated spans so they aren't duplicated in ActivityCompleted.
5. **HALT async bridge** — Hooks may run in sync context but `terminate()` is async. Use a flag that the activity interceptor checks in its `finally` block.
6. **`attribute_key_identifiers`** — Include OTel-based dedup keys per span type so the server can discard duplicate spans.
