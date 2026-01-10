# OpenBox SDK for Temporal Workflows

OpenBox SDK provides governance and observability for Temporal workflows by capturing workflow/activity lifecycle events and HTTP telemetry, then sending them to OpenBox Core for policy evaluation.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Temporal Worker                               │
│                                                                         │
│  ┌────────────────────────┐      ┌────────────────────────────────────┐ │
│  │  Workflow Interceptor  │      │       Activity Interceptor         │ │
│  │  ────────────────────  │      │  ────────────────────────────────  │ │
│  │  - WorkflowStarted     │      │  - ActivityStarted (+ input)       │ │
│  │  - WorkflowCompleted   │      │  - ActivityCompleted (+ output)    │ │
│  │  - WorkflowFailed      │      │                                    │ │
│  │  - SignalReceived      │      │  Guardrails: Redact/modify input   │ │
│  │                        │      │  before execution, output after    │ │
│  │  Sends via activity    │      │                                    │ │
│  │  (determinism)         │      │  Collects HTTP spans with bodies   │ │
│  └────────────────────────┘      └────────────────────────────────────┘ │
│              │                                    │                     │
│              │                                    ▼                     │
│              │         ┌──────────────────────────────────────────────┐ │
│              │         │            WorkflowSpanProcessor             │ │
│              │         │  ──────────────────────────────────────────  │ │
│              │         │  - Buffers spans per workflow                │ │
│              │         │  - Merges body/header data from HTTP hooks   │ │
│              │         │  - Maps trace_id → workflow_id               │ │
│              │         │  - Ignores OpenBox Core URLs                 │ │
│              │         └──────────────────────────────────────────────┘ │
│              │                                    │                     │
│              ▼                                    ▼                     │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                   OTel HTTP Instrumentation                       │  │
│  │  ──────────────────────────────────────────────────────────────  │  │
│  │  - httpx (sync + async with body capture patches)                 │  │
│  │  - requests, urllib3                                              │  │
│  │  Captures: request/response headers + bodies                      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                      ┌─────────────────────────┐
                      │      OpenBox Core       │
                      │   ───────────────────   │
                      │   POST /governance/     │
                      │        evaluate         │
                      │                         │
                      │   Returns:              │
                      │   - verdict: allow/halt │
                      │   - verdict: block      │
                      │   - guardrails_result   │
                      │     (redacted input/    │
                      │      output)            │
                      └─────────────────────────┘
```

## Event Types (6 events)
| Event | Trigger | Key Fields |
|-------|---------|------------|
| `WorkflowStarted` | Workflow begins | workflow_id, run_id, workflow_type, task_queue |
| `WorkflowCompleted` | Workflow succeeds | workflow_id, run_id, workflow_type |
| `WorkflowFailed` | Workflow fails | workflow_id, run_id, workflow_type, **error** |
| `SignalReceived` | Signal received | workflow_id, signal_name, signal_args |
| `ActivityStarted` | Activity begins | activity_id, activity_type, **activity_input** |
| `ActivityCompleted` | Activity ends | activity_id, activity_type, status, **activity_input**, **activity_output**, spans, error |

## Governance Verdicts

OpenBox Core returns a verdict indicating what action the SDK should take.

### v1.1 Verdict Enum (5-tier graduated response)

| Verdict | Value | SDK Behavior |
|---------|-------|--------------|
| `ALLOW` | `"allow"` | Continue execution normally |
| `CONSTRAIN` | `"constrain"` | Log constraints, continue (sandbox enforcement future) |
| `REQUIRE_APPROVAL` | `"require_approval"` | Pause, poll for human approval |
| `BLOCK` | `"block"` | Raise non-retryable error |
| `HALT` | `"halt"` | Raise non-retryable error, terminate workflow |

### Backward Compatibility (v1.0)

The SDK automatically maps v1.0 action strings to v1.1 verdicts:

| v1.0 Action | v1.1 Verdict |
|-------------|--------------|
| `"continue"` | `ALLOW` |
| `"stop"` | `HALT` |
| `"require-approval"` | `REQUIRE_APPROVAL` |

### Verdict Priority

When aggregating multiple verdicts (e.g., from multiple policies), the highest priority wins:

```
HALT (5) > BLOCK (4) > REQUIRE_APPROVAL (3) > CONSTRAIN (2) > ALLOW (1)
```

### v1.1 Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `verdict` | `string` | v1.1 verdict value (see table above) |
| `action` | `string` | v1.0 action (for backward compat) |
| `reason` | `string` | Human-readable explanation |
| `policy_id` | `string` | Policy that triggered the verdict |
| `risk_score` | `float` | Risk score (0.0 - 1.0) |
| `trust_tier` | `string` | Trust tier (v1.1) |
| `alignment_score` | `float` | Alignment score (v1.1) |
| `behavioral_violations` | `array` | List of violations (v1.1) |
| `approval_id` | `string` | Approval tracking ID (v1.1) |
| `constraints` | `array` | Constraints to apply (v1.1) |
| `guardrails_result` | `object` | Guardrails redaction result |

## Guardrails (Input/Output Validation & Redaction)

OpenBox Core can return `guardrails_result` to validate and modify activity input before execution or output after completion:

```json
{
  "verdict": "allow",
  "guardrails_result": {
    "input_type": "activity_input",
    "redacted_input": {"prompt": "[REDACTED]", "user_id": "123"},
    "raw_logs": {"evaluation_id": "eval-123", "model": "gpt-4"},
    "validation_passed": true,
    "reasons": []
  }
}
```

### Guardrails Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `input_type` | `string` | `"activity_input"` or `"activity_output"` |
| `redacted_input` | `any` | Redacted/modified data to replace original |
| `raw_logs` | `object` | Raw logs from guardrails evaluation |
| `validation_passed` | `bool` | If `false`, workflow is terminated |
| `reasons` | `array` | Validation failure reasons (see below) |

### Validation Failure

When `validation_passed` is `false`, the workflow is terminated with a non-retryable `ApplicationError` of type `GuardrailsValidationFailed`. The `reasons` array contains structured failure details:

```json
{
  "verdict": "allow",
  "guardrails_result": {
    "input_type": "activity_input",
    "validation_passed": false,
    "reasons": [
      {"type": "pii", "field": "email", "reason": "Contains PII data"},
      {"type": "sensitive", "field": "ssn", "reason": "SSN detected in input"}
    ]
  }
}
```

Each reason object:
| Field | Type | Description |
|-------|------|-------------|
| `type` | `string` | Category of validation failure |
| `field` | `string` | Field that triggered the failure |
| `reason` | `string` | Human-readable explanation |

### Redaction Behavior

| `input_type` | When Applied | Effect |
|--------------|--------------|--------|
| `activity_input` | Before activity executes | Replaces activity input with redacted version |
| `activity_output` | After activity completes | Replaces activity output with redacted version |

This allows governance to:
1. **Validate** - Block workflows that violate policies (PII, sensitive data, etc.)
2. **Redact** - Sanitize sensitive data without stopping the workflow

## Error Handling Policy

Configure via `on_api_error` in `GovernanceConfig`:

| Policy | Behavior |
|--------|----------|
| `fail_open` (default) | If governance API fails, allow workflow to continue |
| `fail_closed` | If governance API fails, terminate workflow |

## SDK Components

| File | Purpose |
|------|---------|
| `worker.py` | `create_openbox_worker()` - **Recommended** factory for worker-side governance |
| `workflow_interceptor.py` | `GovernanceInterceptor` - workflow lifecycle events (via activity for determinism) |
| `activity_interceptor.py` | `ActivityGovernanceInterceptor` - activity lifecycle with input/output capture, guardrails, and span collection |
| `activities.py` | `send_governance_event` activity for workflow-level HTTP calls |
| `span_processor.py` | `WorkflowSpanProcessor` - buffers spans per workflow_id, merges body/header data |
| `otel_setup.py` | HTTP instrumentation with body/header capture hooks |
| `config.py` | `GovernanceConfig` configuration options |
| `types.py` | `WorkflowEventType` enum, `WorkflowSpanBuffer`, `GuardrailsCheckResult` |

## Quick Start (Recommended)

Use the `create_openbox_worker()` factory function for simple integration:

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

The factory function automatically:
1. Validates the API key
2. Creates WorkflowSpanProcessor
3. Sets up OpenTelemetry HTTP instrumentation
4. Creates governance interceptors (workflow + activity)
5. Adds `send_governance_event` activity
6. Returns a fully configured Worker

### All Parameters

```python
worker = create_openbox_worker(
    client=client,
    task_queue="my-task-queue",
    workflows=[MyWorkflow],
    activities=[my_activity],

    # OpenBox config (required for governance)
    openbox_url="http://localhost:8086",
    openbox_api_key="obx_test_key_1",
    governance_timeout=30.0,           # default: 30.0
    governance_policy="fail_closed",   # default: "fail_open"

    # Event filtering
    send_start_event=True,
    send_activity_start_event=True,
    skip_workflow_types={"InternalWorkflow"},
    skip_activity_types={"send_governance_event"},
    skip_signals={"heartbeat"},

    # Database instrumentation
    instrument_databases=True,         # default: True
    db_libraries={"psycopg2", "redis"}, # default: None (all available)

    # File I/O instrumentation
    instrument_file_io=True,           # default: False

    # Standard Worker options (all supported)
    activity_executor=my_executor,
    max_concurrent_activities=10,
    # ... any other Worker parameter
)
```

## Advanced Usage

For fine-grained control, you can configure components manually:

```python
from temporalio.worker import Worker
from openbox import (
    initialize,
    WorkflowSpanProcessor,
    GovernanceInterceptor,
    GovernanceConfig,
)
from openbox.otel_setup import setup_opentelemetry_for_governance
from openbox.activity_interceptor import ActivityGovernanceInterceptor
from openbox.activities import send_governance_event

# Configuration
openbox_url = "http://localhost:8086"
openbox_key = "obx_test_key_1"

# 1. Initialize SDK (validates API key)
initialize(api_url=openbox_url, api_key=openbox_key)

# 2. Create span processor (ignore OpenBox API calls)
span_processor = WorkflowSpanProcessor(ignored_url_prefixes=[openbox_url])

# 3. Setup OTel instrumentation with body/header capture
setup_opentelemetry_for_governance(span_processor, ignored_urls=[openbox_url])

# 4. Create governance config
config = GovernanceConfig(
    on_api_error="fail_closed",        # or "fail_open"
    api_timeout=30.0,
    send_start_event=True,
    send_activity_start_event=True,
    skip_workflow_types={"InternalWorkflow"},
    skip_activity_types={"send_governance_event"},  # Default
)

# 5. Create interceptors
workflow_interceptor = GovernanceInterceptor(
    api_url=openbox_url,
    api_key=openbox_key,
    span_processor=span_processor,
    config=config,
)

activity_interceptor = ActivityGovernanceInterceptor(
    api_url=openbox_url,
    api_key=openbox_key,
    span_processor=span_processor,
    config=config,
)

# 6. Create worker with both interceptors
worker = Worker(
    client=client,
    task_queue="my-task-queue",
    workflows=[MyWorkflow],
    activities=[my_activity, send_governance_event],  # Include governance activity!
    interceptors=[workflow_interceptor, activity_interceptor],
)
```

## Event Payloads

### WorkflowStarted
```json
{
  "source": "workflow-telemetry",
  "event_type": "WorkflowStarted",
  "workflow_id": "my-workflow-123",
  "run_id": "abc-123",
  "workflow_type": "MyWorkflow",
  "task_queue": "my-task-queue",
  "timestamp": "2024-01-01T00:00:00.000Z"
}
```

### WorkflowFailed
```json
{
  "source": "workflow-telemetry",
  "event_type": "WorkflowFailed",
  "workflow_id": "my-workflow-123",
  "run_id": "abc-123",
  "workflow_type": "MyWorkflow",
  "error": {
    "type": "ApplicationError",
    "message": "Governance blocked: Policy violation detected"
  },
  "timestamp": "2024-01-01T00:00:00.000Z"
}
```

### ActivityCompleted (with input/output and spans)
```json
{
  "source": "workflow-telemetry",
  "event_type": "ActivityCompleted",
  "workflow_id": "my-workflow-123",
  "activity_id": "1",
  "activity_type": "call_llm",
  "status": "completed",
  "duration_ms": 1234.56,
  "activity_input": [{"prompt": "Hello, how are you?"}],
  "activity_output": {"response": "I'm doing well, thank you!"},
  "span_count": 1,
  "spans": [
    {
      "span_id": "abc123",
      "trace_id": "def456",
      "name": "POST",
      "kind": "CLIENT",
      "attributes": {
        "http.method": "POST",
        "http.url": "https://api.openai.com/v1/chat/completions",
        "http.status_code": 200
      },
      "request_body": "{\"model\":\"gpt-4\",\"messages\":[...]}",
      "response_body": "{\"choices\":[...]}"
    }
  ],
  "timestamp": "2024-01-01T00:00:00.000Z"
}
```

### Governance Stop Response
When OpenBox Core returns `verdict: "block"` or `verdict: "halt"` (or v1.0 `action: "stop"`):
```json
{
  "verdict": "halt",
  "reason": "Policy violation: unauthorized API call detected",
  "policy_id": "policy-123",
  "risk_score": 0.95,
  "trust_tier": "untrusted",
  "behavioral_violations": ["unauthorized_api_call"]
}
```

The workflow/activity will be terminated with a non-retryable `ApplicationError`.

### Error Types

| Error Type | Trigger | Description |
|------------|---------|-------------|
| `GovernanceStop` | `verdict: BLOCK/HALT` | Governance policy blocked the workflow |
| `GuardrailsValidationFailed` | `validation_passed: false` | Guardrails validation failed (PII, sensitive data, etc.) |

## Span Data Structures

OpenBox captures different types of spans, each with specific attributes. All spans share a common base structure.

### Base Span Structure

All spans include these core fields:

```json
{
  "span_id": "1a2b3c4d5e6f7890",
  "trace_id": "1a2b3c4d5e6f78901a2b3c4d5e6f7890",
  "parent_span_id": "0987654321fedcba",
  "name": "POST",
  "kind": "CLIENT",
  "start_time": 1704067200000000000,
  "end_time": 1704067201000000000,
  "duration_ns": 1000000000,
  "status": {
    "code": "OK",
    "description": null
  },
  "events": [],
  "attributes": {},
  "activity_id": "1"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `span_id` | `string` | 16-char hex span identifier |
| `trace_id` | `string` | 32-char hex trace identifier |
| `parent_span_id` | `string?` | Parent span ID (null for root spans) |
| `name` | `string` | Span name (e.g., "POST", "SELECT", "file.read") |
| `kind` | `string` | Span kind: `CLIENT`, `SERVER`, `INTERNAL`, `PRODUCER`, `CONSUMER` |
| `start_time` | `int64` | Start time in nanoseconds (Unix epoch) |
| `end_time` | `int64` | End time in nanoseconds |
| `duration_ns` | `int64` | Duration in nanoseconds |
| `status.code` | `string` | `OK`, `ERROR`, or `UNSET` |
| `status.description` | `string?` | Error description if status is ERROR |
| `events` | `array` | Span events (exceptions, logs) |
| `attributes` | `object` | Type-specific attributes (see below) |
| `activity_id` | `string?` | Temporal activity ID (for filtering) |

### HTTP Span Attributes

HTTP spans include request/response bodies and headers in addition to standard OTel attributes:

```json
{
  "attributes": {
    "http.method": "POST",
    "http.url": "https://api.openai.com/v1/chat/completions",
    "http.host": "api.openai.com",
    "http.scheme": "https",
    "http.status_code": 200,
    "http.target": "/v1/chat/completions",
    "net.peer.name": "api.openai.com",
    "net.peer.port": 443
  },
  "request_body": "{\"model\":\"gpt-4\",\"messages\":[...]}",
  "response_body": "{\"choices\":[{\"message\":{...}}]}",
  "request_headers": {
    "content-type": "application/json",
    "authorization": "Bearer sk-..."
  },
  "response_headers": {
    "content-type": "application/json",
    "x-request-id": "req-123"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `http.method` | `string` | HTTP method (GET, POST, PUT, DELETE, etc.) |
| `http.url` | `string` | Full request URL |
| `http.status_code` | `int` | HTTP response status code |
| `http.target` | `string` | Request path and query string |
| `net.peer.name` | `string` | Remote host name |
| `net.peer.port` | `int` | Remote port |
| `request_body` | `string?` | HTTP request body (text content types only) |
| `response_body` | `string?` | HTTP response body (text content types only) |
| `request_headers` | `object?` | HTTP request headers |
| `response_headers` | `object?` | HTTP response headers |

**Note:** Binary content types are not captured. Only text-based content types are included: `text/*`, `application/json`, `application/xml`, `application/javascript`, `application/x-www-form-urlencoded`.

### Database Span Attributes

Database spans capture query information:

```json
{
  "name": "SELECT",
  "attributes": {
    "db.system": "postgresql",
    "db.name": "mydb",
    "db.user": "postgres",
    "db.statement": "SELECT * FROM users WHERE id = $1",
    "db.operation": "SELECT",
    "net.peer.name": "localhost",
    "net.peer.port": 5432
  }
}
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `db.system` | `string` | Database type: `postgresql`, `mysql`, `mongodb`, `redis` |
| `db.name` | `string` | Database name |
| `db.user` | `string` | Database user |
| `db.statement` | `string` | SQL query or command |
| `db.operation` | `string` | Operation type: `SELECT`, `INSERT`, `UPDATE`, `DELETE`, `GET`, `SET` |
| `net.peer.name` | `string` | Database host |
| `net.peer.port` | `int` | Database port |

**MongoDB-specific:**
| Attribute | Description |
|-----------|-------------|
| `db.mongodb.collection` | Collection name |

**Redis-specific:**
| Attribute | Description |
|-----------|-------------|
| `db.redis.database_index` | Redis database index |

### File I/O Span Attributes

File operations are captured as nested spans:

```json
{
  "name": "file.open",
  "attributes": {
    "file.path": "/app/data/config.json",
    "file.mode": "r",
    "file.total_bytes_read": 1024,
    "file.total_bytes_written": 0
  }
}
```

**file.open span:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `file.path` | `string` | Absolute file path |
| `file.mode` | `string` | Open mode: `r`, `w`, `a`, `rb`, `wb`, etc. |
| `file.total_bytes_read` | `int` | Total bytes read (set on close) |
| `file.total_bytes_written` | `int` | Total bytes written (set on close) |

**file.read / file.write child spans:**
```json
{
  "name": "file.read",
  "attributes": {
    "file.path": "/app/data/config.json",
    "file.operation": "read",
    "file.bytes": 512
  }
}
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `file.path` | `string` | File path |
| `file.operation` | `string` | `read`, `readline`, `readlines`, `write`, `writelines` |
| `file.bytes` | `int` | Bytes read/written in this operation |
| `file.lines` | `int` | Line count (for `readlines`/`writelines`) |

**Error attributes (on failure):**
| Attribute | Description |
|-----------|-------------|
| `error` | `true` if operation failed |
| `error.type` | Exception class name (e.g., `FileNotFoundError`) |
| `error.message` | Exception message |

### Internal Function Call Attributes (`@traced`)

Functions decorated with `@traced` create spans with:

```json
{
  "name": "process_data",
  "attributes": {
    "code.function": "process_data",
    "code.namespace": "myapp.processing",
    "function.arg.0": "{\"input\": \"data\"}",
    "function.kwarg.verbose": "true",
    "function.result": "{\"output\": \"processed\"}"
  }
}
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `code.function` | `string` | Function name |
| `code.namespace` | `string` | Module path |
| `function.arg.N` | `string` | Positional argument at index N (JSON serialized) |
| `function.kwarg.X` | `string` | Keyword argument named X (JSON serialized) |
| `function.result` | `string` | Return value (JSON serialized, if `capture_result=True`) |

**Error attributes (on exception):**
| Attribute | Description |
|-----------|-------------|
| `error` | `true` |
| `error.type` | Exception class name |
| `error.message` | Exception message |

**Note:** Arguments and results are truncated at 2000 characters by default. Configure with `max_arg_length` parameter.

## Instrumentation Setup

This section explains how to enable each type of span capture with the OpenBox SDK.

### Quick Setup (All Instrumentation)

The simplest way to enable all instrumentation:

```python
from openbox import create_openbox_worker

worker = create_openbox_worker(
    client=client,
    task_queue="my-queue",
    workflows=[MyWorkflow],
    activities=[my_activity],

    # OpenBox config (required)
    openbox_url=os.getenv("OPENBOX_URL"),
    openbox_api_key=os.getenv("OPENBOX_API_KEY"),

    # Instrumentation options
    instrument_databases=True,   # Capture database queries (default: True)
    instrument_file_io=True,     # Capture file operations (default: False)
)
```

### HTTP Instrumentation (Auto-enabled)

HTTP instrumentation is **automatically enabled** when using `create_openbox_worker()`. No additional setup required.

**Supported libraries:**
| Library | Package | Notes |
|---------|---------|-------|
| httpx | `opentelemetry-instrumentation-httpx` | Sync + async, body capture via patching |
| requests | `opentelemetry-instrumentation-requests` | Full body capture |
| urllib3 | `opentelemetry-instrumentation-urllib3` | Full body capture |
| urllib | `opentelemetry-instrumentation-urllib` | Request body only |

**Required packages** (install if not present):
```bash
uv add opentelemetry-instrumentation-httpx
uv add opentelemetry-instrumentation-requests
uv add opentelemetry-instrumentation-urllib3
```

### Database Instrumentation

Database instrumentation is **enabled by default** but requires the corresponding OTel instrumentation package.

**Supported Databases:**

| Database | Driver Library | OTel Instrumentation Package | `db_libraries` key |
|----------|---------------|------------------------------|-------------------|
| PostgreSQL | `psycopg2` / `psycopg2-binary` | `opentelemetry-instrumentation-psycopg2` | `"psycopg2"` |
| PostgreSQL (async) | `asyncpg` | `opentelemetry-instrumentation-asyncpg` | `"asyncpg"` |
| MySQL | `mysql-connector-python` | `opentelemetry-instrumentation-mysql` | `"mysql"` |
| MySQL | `pymysql` | `opentelemetry-instrumentation-pymysql` | `"pymysql"` |
| MongoDB | `pymongo` | `opentelemetry-instrumentation-pymongo` | `"pymongo"` |
| Redis | `redis` | `opentelemetry-instrumentation-redis` | `"redis"` |
| SQLAlchemy (ORM) | `sqlalchemy` | `opentelemetry-instrumentation-sqlalchemy` | `"sqlalchemy"` |

**Captured Span Attributes by Database:**

| Database | `db.system` | `db.statement` | `db.operation` | Extra Attributes |
|----------|-------------|----------------|----------------|------------------|
| PostgreSQL | `postgresql` | SQL query | `SELECT`, `INSERT`, etc. | `db.name`, `db.user` |
| MySQL | `mysql` | SQL query | `SELECT`, `INSERT`, etc. | `db.name`, `db.user` |
| MongoDB | `mongodb` | Command JSON | `find`, `insert`, etc. | `db.mongodb.collection` |
| Redis | `redis` | Command | `GET`, `SET`, `HGET`, etc. | `db.redis.database_index` |
| SQLAlchemy | varies | SQL query | `SELECT`, `INSERT`, etc. | `db.name` |

**Step 1: Install the instrumentation package for your database:**

```bash
# PostgreSQL (sync)
uv add psycopg2-binary opentelemetry-instrumentation-psycopg2

# PostgreSQL (async)
uv add asyncpg opentelemetry-instrumentation-asyncpg

# MySQL
uv add mysql-connector-python opentelemetry-instrumentation-mysql

# PyMySQL
uv add pymysql opentelemetry-instrumentation-pymysql

# MongoDB
uv add pymongo opentelemetry-instrumentation-pymongo

# Redis
uv add redis opentelemetry-instrumentation-redis

# SQLAlchemy ORM
uv add sqlalchemy opentelemetry-instrumentation-sqlalchemy
```

**Step 2: Configure the worker:**

```python
# Option A: Instrument all available databases (default)
worker = create_openbox_worker(
    ...,
    instrument_databases=True,  # Default
)

# Option B: Instrument specific databases only
worker = create_openbox_worker(
    ...,
    db_libraries={"psycopg2", "redis"},  # Only these
)

# Option C: Disable database instrumentation
worker = create_openbox_worker(
    ...,
    instrument_databases=False,
)
```

**Example: Capturing PostgreSQL queries**

```python
import psycopg2

# This query will be captured as a span
conn = psycopg2.connect("postgresql://user:pass@localhost/mydb")
cursor = conn.cursor()
cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
row = cursor.fetchone()
cursor.close()
conn.close()
```

The span will include:
```json
{
  "name": "SELECT",
  "attributes": {
    "db.system": "postgresql",
    "db.name": "mydb",
    "db.statement": "SELECT * FROM users WHERE id = %s",
    "db.operation": "SELECT"
  }
}
```

### File I/O Instrumentation

File I/O instrumentation is **disabled by default** (can be noisy). Enable it explicitly:

```python
worker = create_openbox_worker(
    ...,
    instrument_file_io=True,
)
```

**Example: Capturing file operations**

```python
# These operations will be captured as spans
with open("/app/config.json", "r") as f:
    content = f.read()  # Creates file.read span

with open("/app/output.txt", "w") as f:
    f.write("Hello, World!")  # Creates file.write span
```

**Skipped paths:** System paths are automatically ignored to reduce noise:
- `/dev/`, `/proc/`, `/sys/`
- `__pycache__`, `.pyc`, `.pyo`, `.so`, `.dylib`

### Internal Function Tracing (`@traced`)

Use the `@traced` decorator to capture custom function calls:

**Step 1: Import the decorator**

```python
from openbox.tracing import traced
```

**Step 2: Decorate functions to trace**

```python
@traced
def process_payment(order_id: str, amount: float) -> dict:
    # Business logic here
    return {"status": "success", "transaction_id": "txn_123"}

@traced
async def fetch_user_data(user_id: str) -> dict:
    # Async functions work too
    return await db.get_user(user_id)
```

**Step 3: Configure capture options (optional)**

```python
# Capture arguments and results (default)
@traced(capture_args=True, capture_result=True)
def my_function(data):
    return process(data)

# Don't capture sensitive results
@traced(capture_result=False)
def handle_password(password: str) -> bool:
    return verify(password)

# Custom span name
@traced(name="payment-processing")
def process_payment(order):
    return charge(order)

# Limit argument size (default: 2000 chars)
@traced(max_arg_length=500)
def handle_large_input(big_data):
    return summarize(big_data)
```

**Manual span creation** (for fine-grained control):

```python
from openbox.tracing import create_span

def complex_operation(data):
    with create_span("validate-input", {"data_size": len(data)}) as span:
        validated = validate(data)
        span.set_attribute("validation.passed", True)

    with create_span("transform-data") as span:
        result = transform(validated)
        span.set_attribute("output_size", len(result))

    return result
```

### Advanced: Manual Setup

For fine-grained control, set up instrumentation manually:

```python
from openbox.span_processor import WorkflowSpanProcessor
from openbox.otel_setup import setup_opentelemetry_for_governance

# Create span processor
span_processor = WorkflowSpanProcessor(
    ignored_url_prefixes=["http://localhost:8086"]  # Ignore OpenBox API
)

# Setup instrumentation
setup_opentelemetry_for_governance(
    span_processor=span_processor,
    ignored_urls=["http://localhost:8086"],

    # Database options
    instrument_databases=True,
    db_libraries={"psycopg2", "asyncpg", "redis"},  # Or None for all

    # File I/O options
    instrument_file_io=True,
)
```

### Troubleshooting

**Database queries not captured?**

1. Check if the OTel instrumentation package is installed:
   ```bash
   uv pip list | grep instrumentation
   ```

2. Ensure instrumentation is enabled BEFORE database connections are created:
   ```python
   # WRONG: Database imported before worker setup
   import psycopg2  # Module-level import

   worker = create_openbox_worker(...)  # Too late!

   # RIGHT: Worker setup happens at application start
   # (before any database operations)
   ```

3. Verify OpenBox is configured:
   ```bash
   # These must be set
   echo $OPENBOX_URL
   echo $OPENBOX_API_KEY
   ```

**File I/O spans missing?**

1. Ensure `instrument_file_io=True` is set
2. Check if the path is in the skip list (system paths are ignored)

**HTTP spans missing bodies?**

1. Only text content types are captured (not binary)
2. Check if the URL is in the ignored list
3. Ensure httpx/requests instrumentation packages are installed

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `on_api_error` | `"fail_open"` | `"fail_open"` = continue on API error, `"fail_closed"` = stop on API error |
| `api_timeout` | `30.0` | HTTP timeout for governance API calls (seconds) |
| `send_start_event` | `True` | Send WorkflowStarted events |
| `send_activity_start_event` | `True` | Send ActivityStarted events (with input) |
| `skip_workflow_types` | `set()` | Workflow types to skip |
| `skip_activity_types` | `{"send_governance_event"}` | Activity types to skip |
| `skip_signals` | `set()` | Signal names to skip |

## Environment Variables (Example)

Configure these in your `.env` file and pass to `create_openbox_worker()`:

```bash
OPENBOX_URL=http://localhost:8086
OPENBOX_API_KEY=obx_test_key_1
OPENBOX_GOVERNANCE_TIMEOUT=30.0
OPENBOX_GOVERNANCE_POLICY=fail_closed  # fail_open or fail_closed
```

```python
# In your worker code
worker = create_openbox_worker(
    ...,
    openbox_url=os.getenv("OPENBOX_URL"),
    openbox_api_key=os.getenv("OPENBOX_API_KEY"),
    governance_timeout=float(os.getenv("OPENBOX_GOVERNANCE_TIMEOUT", "30.0")),
    governance_policy=os.getenv("OPENBOX_GOVERNANCE_POLICY", "fail_open"),
)
```

## Key Design Decisions

1. **Workflow determinism**: Workflow interceptor sends events via `send_governance_event` activity because workflows cannot make HTTP calls directly.

2. **Activity direct HTTP**: Activity interceptor sends events directly since activities are allowed to make HTTP calls.

3. **Input/Output capture**: Activity arguments and return values are serialized and included in governance events for policy evaluation.

4. **Body/header capture**: Stored separately from OTel span attributes to keep sensitive data out of external tracing systems.

5. **trace_id mapping**: Child HTTP spans are associated with parent activity via trace_id → workflow_id/activity_id mapping.

6. **Governance stop**: When API returns `verdict: "block"` or `verdict: "halt"`, raises `ApplicationError` with `non_retryable=True` to immediately terminate the workflow.

7. **Fail-open/closed policy**: Configurable behavior when governance API is unreachable.

## Function Tracing

OpenBox SDK provides a `@traced` decorator to capture internal function calls as spans. These spans are automatically included in governance events.

### Basic Usage

```python
from openbox.tracing import traced

@traced
def process_data(input_data):
    return transform(input_data)

@traced
async def fetch_external_data(url):
    return await http_get(url)
```

### With Options

```python
from openbox.tracing import traced

@traced(name="custom-span-name", capture_args=True, capture_result=True)
def my_function(data):
    return process(data)

# Don't capture sensitive results
@traced(capture_result=False)
def handle_credentials(username, password):
    return authenticate(username, password)
```

### Manual Span Creation

```python
from openbox.tracing import create_span

def complex_operation(data):
    with create_span("step-1", {"input": data}) as span:
        result = do_step_1(data)
        span.set_attribute("step1.result", result)

    with create_span("step-2") as span:
        final = do_step_2(result)

    return final
```

### Span Attributes

Traced functions include these attributes:
| Attribute | Description |
|-----------|-------------|
| `code.function` | Function name |
| `code.namespace` | Module name |
| `function.arg.N` | Positional arguments (if `capture_args=True`) |
| `function.kwarg.X` | Keyword arguments (if `capture_args=True`) |
| `function.result` | Return value (if `capture_result=True`) |
| `error` | `True` if exception occurred |
| `error.type` | Exception class name |
| `error.message` | Exception message |

## File I/O Instrumentation

OpenBox SDK can capture file read/write operations as spans.

### Enabling File I/O Instrumentation

```python
worker = create_openbox_worker(
    ...,
    instrument_file_io=True,
)
```

### Captured Operations

| Operation | Span Name | Attributes |
|-----------|-----------|------------|
| `open(path, mode)` | `file.open` | `file.path`, `file.mode` |
| `file.read()` | `file.read` | `file.path`, `file.bytes` |
| `file.write(data)` | `file.write` | `file.path`, `file.bytes` |
| `file.readline()` | `file.readline` | `file.path`, `file.bytes` |
| `file.readlines()` | `file.readlines` | `file.path`, `file.lines`, `file.bytes` |

### Skipped Paths

System paths are automatically skipped: `/dev/`, `/proc/`, `/sys/`, `__pycache__`, `.pyc`, `.so`

## Database Instrumentation

OpenBox SDK can capture database queries as spans, enabling governance policies on database operations.

### Supported Databases

| Database | Library | OTel Package |
|----------|---------|--------------|
| PostgreSQL | psycopg2 | `opentelemetry-instrumentation-psycopg2` |
| PostgreSQL (async) | asyncpg | `opentelemetry-instrumentation-asyncpg` |
| MySQL | mysql-connector-python | `opentelemetry-instrumentation-mysql` |
| MySQL | pymysql | `opentelemetry-instrumentation-pymysql` |
| MongoDB | pymongo | `opentelemetry-instrumentation-pymongo` |
| Redis | redis | `opentelemetry-instrumentation-redis` |
| SQLAlchemy | sqlalchemy | `opentelemetry-instrumentation-sqlalchemy` |

### Enabling Database Instrumentation

Database instrumentation is **enabled by default**. Install the OTel instrumentation package for your database:

```bash
# PostgreSQL
pip install opentelemetry-instrumentation-psycopg2

# Or for async PostgreSQL
pip install opentelemetry-instrumentation-asyncpg

# MongoDB
pip install opentelemetry-instrumentation-pymongo

# Redis
pip install opentelemetry-instrumentation-redis
```

### Configuration

```python
# Default: instrument all available databases
worker = create_openbox_worker(
    client=client,
    task_queue="my-queue",
    workflows=[MyWorkflow],
    activities=[my_activity],
    openbox_url=os.getenv("OPENBOX_URL"),
    openbox_api_key=os.getenv("OPENBOX_API_KEY"),
    # instrument_databases=True (default)
)

# Disable database instrumentation
worker = create_openbox_worker(
    ...,
    instrument_databases=False,
)

# Instrument only specific databases
worker = create_openbox_worker(
    ...,
    db_libraries={"psycopg2", "redis"},
)
```

### Database Span Data

Database spans include:

| Attribute | Description |
|-----------|-------------|
| `db.system` | Database type (postgresql, mysql, mongodb, redis) |
| `db.name` | Database name |
| `db.statement` | SQL query or command |
| `db.operation` | Operation type (SELECT, INSERT, GET, etc.) |
| `net.peer.name` | Database host |
| `net.peer.port` | Database port |

**Note:** Unlike HTTP spans, database spans do not capture query results (only the query itself).

## Requirements

- Python 3.9+
- temporalio
- opentelemetry-api
- opentelemetry-sdk
- opentelemetry-instrumentation-httpx
- httpx

### Optional Database Instrumentation Packages

```bash
pip install opentelemetry-instrumentation-psycopg2  # PostgreSQL
pip install opentelemetry-instrumentation-asyncpg   # PostgreSQL async
pip install opentelemetry-instrumentation-mysql     # MySQL
pip install opentelemetry-instrumentation-pymysql   # PyMySQL
pip install opentelemetry-instrumentation-pymongo   # MongoDB
pip install opentelemetry-instrumentation-redis     # Redis
pip install opentelemetry-instrumentation-sqlalchemy # SQLAlchemy ORM
```