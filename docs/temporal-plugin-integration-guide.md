# OpenBox Plugin Integration Guide

**For:** Temporal AI Partner Ecosystem
**Version:** 1.2.0
**Requires:** `temporalio >= 1.24.0`

---

## What is OpenBox?

OpenBox provides governance and observability for Temporal workflows. It captures workflow/activity lifecycle events, HTTP requests, database queries, and file operations, then sends them to OpenBox Core for real-time policy evaluation. Verdicts (ALLOW, BLOCK, HALT, REQUIRE_APPROVAL, CONSTRAIN) control whether operations proceed.

---

## Installation

```bash
pip install openbox-temporal-sdk-python
```

---

## Quick Start

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

---

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `openbox_url` | str | required | OpenBox Core API URL |
| `openbox_api_key` | str | required | API key (`obx_live_*` or `obx_test_*`) |
| `governance_timeout` | float | 30.0 | Timeout for governance API calls (seconds) |
| `governance_policy` | str | `"fail_open"` | `"fail_open"` or `"fail_closed"` |
| `send_start_event` | bool | True | Send WorkflowStarted events |
| `send_activity_start_event` | bool | True | Send ActivityStarted events |
| `skip_workflow_types` | Set[str] | None | Workflow types to skip governance |
| `skip_activity_types` | Set[str] | None | Activity types to skip (default: `send_governance_event`) |
| `skip_signals` | Set[str] | None | Signal names to skip governance |
| `hitl_enabled` | bool | True | Enable human-in-the-loop approval polling |
| `instrument_databases` | bool | True | Instrument database libraries |
| `db_libraries` | set | None | Specific DB libraries to instrument (None = all) |
| `sqlalchemy_engine` | Any | None | Pre-existing SQLAlchemy engine to instrument |
| `instrument_file_io` | bool | True | Instrument file I/O operations |

---

## How It Works

The plugin composes existing OpenBox SDK components via Temporal's `SimplePlugin` base class:

1. **Constructor** — validates API key, sets up OTel instrumentation, creates governance interceptors
2. **`configure_worker()`** — stores Temporal client reference (for HALT terminate calls), delegates to `SimplePlugin` to append interceptors and activities
3. **`workflow_runner`** — adds sandbox passthrough for `opentelemetry` module

### Interceptor Ordering

Plugin interceptors are appended (innermost). This means user-provided interceptors run first, then OpenBox governance. This is correct for governance — it observes the final state of operations.

---

## Governance Verdicts

| Verdict | Behavior |
|---------|----------|
| `ALLOW` | Continue normally |
| `CONSTRAIN` | Log constraints, continue |
| `REQUIRE_APPROVAL` | Pause, poll for human approval |
| `BLOCK` | Raise error, stop activity |
| `HALT` | Raise error, terminate workflow |

---

## Hook-Level Governance

Every HTTP request, database query, and file operation during an activity is evaluated in real-time:

- **HTTP** — request/response hooks via OTel instrumentation (httpx, requests, urllib3)
- **Database** — per-query hooks (psycopg2, asyncpg, pymysql, pymongo, redis, sqlalchemy)
- **File I/O** — per-operation hooks (open, read, write, close)
- **Function tracing** — `@traced` decorator for governed function calls

Each operation is evaluated at `started` (pre-execution, can block) and `completed` (post-execution).

---

## Human-in-the-Loop

When a `REQUIRE_APPROVAL` verdict is returned:

1. Activity pauses and polls for approval
2. Approval status checked via OpenBox Core API
3. On approval → activity continues
4. On rejection/expiry → activity fails with `ApprovalRejectedError`/`ApprovalExpiredError`

---

## Error Handling

| Policy | Behavior |
|--------|----------|
| `fail_open` (default) | If governance API fails, allow workflow to continue |
| `fail_closed` | If governance API fails, terminate workflow |

---

## Composability

OpenBoxPlugin works alongside other Temporal plugins:

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

It also works with user-provided interceptors — both are active simultaneously.
