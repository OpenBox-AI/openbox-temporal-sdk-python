# System Architecture

**Last Updated:** 2026-03-16
**Version:** 1.1.0
**Total LOC:** 3,700+ (across 11 Python files)

---

## Overview

OpenBox SDK for Temporal Workflows is a governance and observability layer that sits between Temporal workflows and OpenBox Core. It captures workflow/activity lifecycle events, HTTP telemetry, database queries, and file operations, then sends them to OpenBox Core for policy evaluation.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Application                            │
│  ┌──────────────────┐              ┌──────────────────┐            │
│  │   Workflows      │              │   Activities     │            │
│  │  (Deterministic) │              │ (Non-deterministic)│          │
│  └──────────────────┘              └──────────────────┘            │
└────────────┬────────────────────────────────┬─────────────────────┘
             │                                 │
             ▼                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      OpenBox SDK Layer                              │
│  ┌──────────────────────────┐    ┌──────────────────────────────┐  │
│  │ GovernanceInterceptor    │    │ ActivityGovernanceInterceptor│  │
│  │ ──────────────────────   │    │ ────────────────────────────  │  │
│  │ - WorkflowStarted        │    │ - ActivityStarted            │  │
│  │ - WorkflowCompleted      │    │ - ActivityCompleted          │  │
│  │ - WorkflowFailed         │    │ - Input/Output capture       │  │
│  │ - SignalReceived         │    │ - Guardrails enforcement     │  │
│  │                          │    │ - HITL approval polling      │  │
│  │ Sends via activity       │    │ Sends via direct HTTP        │  │
│  └────────────┬─────────────┘    └──────────┬───────────────────┘  │
│               │                               │                      │
│               ▼                               ▼                      │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │            WorkflowSpanProcessor (Span Buffering)              ││
│  │  ────────────────────────────────────────────────────────────  ││
│  │  - Buffer spans per workflow_id                                ││
│  │  - Store HTTP bodies/headers separately (privacy)              ││
│  │  - Map trace_id → workflow_id for child spans                  ││
│  │  - Store verdicts from SignalReceived                          ││
│  └────────────────────────────────────────────────────────────────┘│
│               │                                                      │
│               ▼                                                      │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │ Hook-Level Governance Layer (Real-time per-operation)         ││
│  │  ────────────────────────────────────────────────────────────  ││
│  │ - HTTP request/response hooks evaluate at started/completed    ││
│  │ - Database query hooks evaluate before/after execution         ││
│  │ - File I/O hooks evaluate on open/read/write operations        ││
│  │ - @traced decorator evaluates function calls                   ││
│  └────────────────────────────────────────────────────────────────┘│
│               │                                                      │
│               ▼                                                      │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │         OpenTelemetry Instrumentation Layer                    ││
│  │  ────────────────────────────────────────────────────────────  ││
│  │  HTTP:    httpx, requests, urllib3, urllib                     ││
│  │  Database: PostgreSQL, MySQL, MongoDB, Redis, SQLAlchemy      ││
│  │  File I/O: open(), read(), write()                            ││
│  │  Functions: @traced decorator                                 ││
│  └────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       OpenBox Core API                              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  POST /api/v1/governance/evaluate                            │  │
│  │  POST /api/v1/governance/approval                            │  │
│  │  GET  /api/v1/auth/validate                                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Returns: verdict, reason, guardrails_result, approval status      │
└─────────────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Temporal Server                               │
│  (Workflow orchestration, task queues, history)                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. Interceptor Layer

#### GovernanceInterceptor (Workflow-Level)

**Responsibility:** Capture workflow lifecycle events

**Key Characteristics:**
- Workflow-safe (no HTTP, no datetime, no os.stat)
- Events sent via `send_governance_event` activity for determinism
- Stores BLOCK/HALT verdicts from SignalReceived for activity interceptor

**Event Flow:**
```
1. Workflow starts
   → GovernanceInterceptor.execute_workflow() called
   → Sends WorkflowStarted via activity
   → Executes user workflow code

2. Workflow completes successfully
   → Sends WorkflowCompleted via activity
   → Returns result

3. Workflow fails
   → Extracts exception chain
   → Sends WorkflowFailed via activity
   → Re-raises exception

4. Signal received
   → Sends SignalReceived via activity
   → If verdict is BLOCK/HALT, stores in span processor
   → Next activity will check verdict and fail
```

**Code Location:** `openbox/workflow_interceptor.py`

#### ActivityGovernanceInterceptor (Activity-Level)

**Responsibility:** Capture activity execution with input/output and spans

**Key Characteristics:**
- Activity-only (direct HTTP allowed)
- Captures activity arguments and return values
- Collects child spans (HTTP, database, file I/O)
- Enforces guardrails redaction
- Polls for HITL approval on retry

**Event Flow:**
```
1. Activity starts
   → Check for pending BLOCK/HALT verdict from signal
   → Check for pending approval and poll if present
   → Register workflow buffer if needed
   → Send ActivityStarted event (optional)
   → Apply input guardrails if present

2. Activity executes
   → Create OTel span with trace_id mapping
   → User activity code runs
   → Child spans (HTTP/DB/file) captured automatically
   → Hook-level governance evaluates each operation in real-time

3. Activity completes
   → Collect child spans from buffer
   → Send ActivityCompleted event with input/output/spans
   → Apply output guardrails if present
   → Handle REQUIRE_APPROVAL verdict (retry with polling)

4. Activity retries (if approval pending)
   → Poll /api/v1/governance/approval
   → If approved: clear pending, proceed
   → If rejected: raise non-retryable error
   → If expired: terminate workflow
```

**Code Location:** `openbox/activity_interceptor.py`

---

### 2. Span Buffering Layer

#### WorkflowSpanProcessor

**Responsibility:** Buffer spans per workflow and merge body/header data

**Key Data Structures:**
```python
class WorkflowSpanProcessor:
    _buffers: Dict[str, WorkflowSpanBuffer]          # workflow_id → buffer
    _trace_to_workflow: Dict[int, str]               # trace_id → workflow_id
    _trace_to_activity: Dict[int, str]               # trace_id → activity_id
    _body_data: Dict[int, dict]                      # span_id → {bodies, headers}
    _verdicts: Dict[str, dict]                       # workflow_id → verdict
```

**Span Buffering Flow:**
```
1. Activity starts
   → ActivityInterceptor creates OTel span
   → Calls span_processor.register_trace(trace_id, workflow_id, activity_id)
   → Child spans (HTTP/DB) share same trace_id

2. HTTP call made
   → OTel HTTP instrumentation creates child span
   → Hook captures request/response bodies
   → Calls span_processor.store_body(span_id, request_body=..., response_body=...)

3. Span ends
   → span_processor.on_end(span) called by OTel
   → Looks up workflow_id via span attributes or trace_id mapping
   → Merges body data from _body_data into span dict
   → Appends span to workflow buffer

4. Activity completes
   → ActivityInterceptor retrieves buffer.spans
   → Filters spans by activity_id
   → Sends to OpenBox Core in ActivityCompleted event
```

**Privacy Design:**
- Bodies stored in `_body_data` dict, NOT in OTel span attributes
- Merged into span dict only when sending to OpenBox Core
- Optional fallback OTel processor receives spans WITHOUT bodies

**Code Location:** `openbox/span_processor.py`

---

### 3. Hook-Level Governance Layer

#### Hook Governance Module (`hook_governance.py`)

**Responsibility:** Real-time governance evaluation for each operation

**Architecture:**
1. Hook modules detect an operation and build a `span_data` dict
2. Hook calls `evaluate_sync()` or `evaluate_async()` in `hook_governance.py`
3. This module: looks up activity context, assembles payload, sends to API
4. If verdict is BLOCK/HALT/REQUIRE_APPROVAL → raises `GovernanceBlockedError`

**Payload Structure (hook_trigger simplified to boolean):**
```json
{
  "workflow_id": "string",
  "activity_id": "string",
  "run_id": "string",
  "hook_trigger": true,
  "timestamp": "RFC3339 timestamp",
  "spans": [
    {
      "span_id": "16-char hex",
      "trace_id": "32-char hex",
      "parent_span_id": "16-char hex or null",
      "stage": "started|completed",
      "hook_type": "http_request|db_query|file_operation|function_call",
      "start_time": nanoseconds,
      "end_time": nanoseconds or null,
      "duration_ns": nanoseconds or null,
      "name": "operation name",
      "kind": "CLIENT|INTERNAL|SERVER",
      "attributes": {...OTel-original only...},
      "status": {"code": "ERROR|UNSET", "description": "string or null"},
      "events": [],

      // Type-specific fields at root level (not in attributes)
      // HTTP-specific:
      "http_method": "GET|POST|...",
      "http_url": "string",
      "http_status_code": number,
      "request_body": "string or null",
      "request_headers": {...},
      "response_body": "string or null",
      "response_headers": {...},
      "error": "string or null",

      // Database-specific:
      "db_system": "postgresql|mysql|mongodb|redis",
      "db_operation": "SELECT|INSERT|UPDATE|DELETE|GET|...",
      "db_statement": "SQL or command",
      "db_name": "database name",
      "server_address": "host",
      "server_port": number,

      // File-specific:
      "file_path": "string",
      "file_mode": "r|w|a|...",
      "file_operation": "open|read|write|close",
      "file_bytes": number,

      // Function-specific:
      "code_function": "function name",
      "code_namespace": "module name"
    }
  ],
  "span_count": 1
}
```

**Key Characteristics:**
- `hook_trigger` is simple boolean `true` (replaces dict with type/stage/data)
- All span data at root level: `http_method`, `db_system`, `file_path`, `function` (type-specific fields NOT in attributes)
- `attributes` field contains ONLY original OTel attributes (no custom fields injected)
- `hook_type` field discriminates operation type: `http_request`, `db_query`, `file_operation`, `function_call`
- Single span per evaluation (no accumulated history)
- `duration_ns` computed for all hook types

**Code Location:** `openbox/hook_governance.py`

---

### 4. Instrumentation Layer

#### HTTP Instrumentation

**Supported Libraries:**
- `httpx` - Sync + async, full body capture via Client.send patching
- `requests` - Full body capture via hooks
- `urllib3` - Full body capture via hooks
- `urllib` - Request body only (response stream consumed)

**Instrumentation Strategy:**

1. **OTel Instrumentors** - Create spans with standard attributes
   ```python
   from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
   HTTPXClientInstrumentor().instrument(
       request_hook=_httpx_request_hook,
       response_hook=_httpx_response_hook,
   )
   ```

2. **Hook-Level Governance** (request hook) - started stage governance evaluation
   - Builds span data and calls governance evaluate_sync/evaluate_async
   - Span data includes type-specific fields at root: http_method, http_url, request/response bodies, http_status_code
   - Can block HTTP request if governance returns BLOCK/HALT
   ```python
   def _httpx_request_hook(span, request):
       # Build span data with hook_type and type-specific fields at root level
       span_data = _build_http_span_data(span, method, url, "started", request_body=body)
       # Evaluate governance
       _hook_gov.evaluate_sync(span, identifier=url, span_data=span_data)
   ```

3. **Custom Hooks** - Capture bodies/headers via `span_processor.store_body()`
   ```python
   def _httpx_request_hook(span, request):
       body = extract_body(request)
       headers = dict(request.headers)
       _span_processor.store_body(
           span.context.span_id,
           request_body=body,
           request_headers=headers,
       )
   ```

4. **Response Hook** (completed stage) - Governance evaluation with response
   - Builds span data with response body and status code
   - Calls governance evaluate_sync/evaluate_async with completed stage
   - Status set to "ERROR" if http_status_code >= 400

5. **Client.send Patching** (httpx only) - Reliable body capture
   ```python
   def _patched_async_send(self, request, *args, **kwargs):
       request_body = request.content
       response = await _original_async_send(self, request, *args, **kwargs)
       response_body = response.text
       span_processor.store_body(span_id, request_body=..., response_body=...)
       return response
   ```

6. **Span Data Format** (for governance evaluation)
   ```
   stage: "started" | "completed"
   span_id: hex string (16 chars)
   trace_id: hex string (32 chars)
   parent_span_id: hex string or null
   name: "HTTP {METHOD}" (e.g., "HTTP GET")
   kind: "CLIENT"
   start_time: nanosecond timestamp
   end_time: nanosecond timestamp (completed only), None for started
   status: {code: "ERROR" if status >= 400, "UNSET", description: error}
   hook_type: "http_request"
   attributes: {http.method, http.url, http.status_code} (OTel-original only)

   // Type-specific fields at root level:
   http_method: HTTP method
   http_url: HTTP URL
   http_status_code: HTTP status code
   request_body: request body text
   response_body: response body text
   request_headers: dict of request headers
   response_headers: dict of response headers
   ```

**Code Location:** `openbox/http_governance_hooks.py` (re-exported via `openbox/otel_setup.py`)

#### Database Instrumentation

**Supported Databases:**
- PostgreSQL: `psycopg2` (sync), `asyncpg` (async)
- MySQL: `mysql-connector-python`, `pymysql`
- MongoDB: `pymongo`
- Redis: `redis`
- SQLAlchemy: `sqlalchemy` (ORM)

**Instrumentation Strategy:**

1. **DB Governance Hooks** (installed BEFORE OTel) — per-query started/completed governance
   ```python
   # db_governance_hooks.py installs wrapt/event hooks before OTel instrumentors
   _db_gov.setup_psycopg2_hooks()       # wrapt on cursor.execute
   Psycopg2Instrumentor().instrument()   # OTel span creation
   ```
   - Both started + completed stages call governance evaluate with `span_data=` parameter
   - Span data includes: db.system, db.operation, db.statement, status, duration_ns

2. **OTel Instrumentors** - Create spans with db.* attributes
   ```python
   # Span attributes:
   {
       "db.system": "postgresql",
       "db.statement": "SELECT * FROM users WHERE id = $1",
       "db.operation": "SELECT",
       "db.name": "mydb",
   }
   ```

3. **Span Data Format** (for governance evaluation)
   ```
   stage: "started" | "completed"
   span_id: hex string (16 chars)
   trace_id: hex string (32 chars)
   parent_span_id: hex string or null
   name: "{OPERATION} {SYSTEM}" (e.g., "SELECT postgresql")
   kind: "CLIENT"
   start_time: nanosecond timestamp
   end_time: nanosecond timestamp (completed only), None for started
   status: {code: "ERROR" if error, "UNSET", description: error or null}
   hook_type: "db_query"
   attributes: {db.system, db.operation, db.statement, db.name, server.address, server.port, rowcount}

   // Type-specific fields at root level:
   db_system: "postgresql" etc
   db_operation: "SELECT" etc
   db_statement: SQL query
   db_name: database name
   server_address: host
   server_port: port
   duration_ns: query duration in nanoseconds
   ```

4. **Span Capture** - Automatically buffered by WorkflowSpanProcessor

**Per-library hook strategy:**

| Library | Method | Notes |
|---------|--------|-------|
| redis | Native OTel `request_hook`/`response_hook` | Passed to `RedisInstrumentor().instrument()` |
| sqlalchemy | `before/after_cursor_execute` + `handle_error` events | Requires engine reference |
| psycopg2, asyncpg, mysql, pymysql, pymongo | `wrapt` monkey-patching | C extensions may be immutable (silently skipped) |

**Code Location:** `openbox/db_governance_hooks.py` (orchestrated by `openbox/otel_setup.py`)

#### File I/O Instrumentation

**Implementation:** Monkey-patch `builtins.open` with `TracedFile` wrapper

**Instrumentation Strategy:**

1. **Patch open()** - Replace with tracing wrapper
   ```python
   _original_open = builtins.open

   def traced_open(file, mode='r', *args, **kwargs):
       span = tracer.start_span("file.open")
       file_obj = _original_open(file, mode, *args, **kwargs)
       return TracedFile(file_obj, file_path, mode, span)
   ```

2. **Hook-Level Governance** (on open, read, write) - started/completed stages
   - Builds span data with file path, mode, operation
   - Both started + completed stages call governance evaluate_sync with span_data
   - Can block file access (open, read, write) if governance returns BLOCK/HALT
   ```python
   def _evaluate_governance(self, operation: str, stage: str, span=None):
       span_data = _build_file_span_data(span, file_path, mode, operation, stage)
       _hook_gov.evaluate_sync(span, identifier=file_path, span_data=span_data)
   ```

3. **Wrap File Operations** - Trace each read/write
   ```python
   class TracedFile:
       def read(self, size=-1):
           with tracer.start_as_current_span("file.read") as span:
               # Governance started stage
               self._evaluate_governance("read", "started", span=span)
               data = self._file.read(size)
               span.set_attribute("file.bytes", len(data))
               # Governance completed stage
               self._evaluate_governance("read", "completed", span=span, data=data)
               return data
   ```

4. **Span Data Format** (for governance evaluation)
   ```
   stage: "started" | "completed"
   span_id: hex string (16 chars)
   trace_id: hex string (32 chars)
   parent_span_id: hex string or null
   name: "file.{operation}" (e.g., "file.read")
   kind: "INTERNAL"
   start_time: nanosecond timestamp
   end_time: nanosecond timestamp (completed only), None for started
   status: {code: "ERROR" if error, "UNSET", description: error}
   hook_type: "file_operation"
   attributes: {file.path, file.mode, file.operation, openbox.governance.error}

   // Type-specific fields at root level:
   file_path: path to file
   file_mode: "r", "w", etc
   file_operation: "open", "read", "write"
   file_bytes: bytes read/written
   duration_ns: operation duration
   ```

**Skipped Paths:** `/dev/`, `/proc/`, `/sys/`, `__pycache__`, `.pyc`, `.so`

**Code Location:** `openbox/file_governance_hooks.py` (re-exported via `openbox/otel_setup.py`)

#### Function Tracing Instrumentation

**Implementation:** `@traced` decorator in `tracing.py`

**Decorator Features:**
- Supports sync and async functions
- Creates OTel span with configurable name
- Captures function arguments, return values, exceptions
- Serializes args/results safely with max length limits
- Hook-level governance at started and completed stages

**Instrumentation Strategy:**

1. **Decorator Wrapper** - Wraps function execution
   ```python
   @traced
   def my_function(arg1, arg2):
       return do_something(arg1, arg2)

   @traced(name="custom-name", capture_args=True, capture_result=True)
   async def my_async_function(data):
       return await process(data)
   ```

2. **Hook-Level Governance** - When `hook_governance` is configured
   - `started` stage: Before function executes (can block)
   - `completed` stage: After function returns or raises (can block/halt)
   - Both stages pass `span_data=` parameter with structured span info
   - Zero overhead when governance not configured

3. **Span Data Format** (for governance evaluation)
   ```
   stage: "started" | "completed"
   span_id: hex string (16 chars)
   trace_id: hex string (32 chars)
   parent_span_id: hex string or null
   name: function name or custom span name
   kind: "INTERNAL"
   start_time: nanosecond timestamp
   end_time: nanosecond timestamp (completed only), None for started
   status: {code: "ERROR" | "UNSET", description: error message or null}
   hook_type: "function_call"
   attributes: {code.function, code.namespace, ...function args, result, errors}

   // Type-specific fields at root level:
   code_function: function name
   code_namespace: module name
   ```

4. **Span Attributes**
   ```
   code.function = function name
   code.namespace = module name
   function.arg.N = positional args (JSON)
   function.kwarg.X = keyword args (JSON)
   function.result = return value (JSON)
   error / error.type / error.message = exception details
   ```

**Code Location:** `openbox/tracing.py`

---

### 5. Governance Integration Layer

#### OpenBox Core API

**Base URL:** Configurable (e.g., `http://localhost:8086`)

**Endpoints:**

##### POST /api/v1/governance/evaluate
**Purpose:** Evaluate governance event, return verdict

**Request Schema:**
```typescript
interface GovernanceEvent {
  source: "workflow-telemetry";
  event_type: "WorkflowStarted" | "WorkflowCompleted" | "WorkflowFailed" |
              "SignalReceived" | "ActivityStarted" | "ActivityCompleted";
  workflow_id: string;
  run_id: string;
  workflow_type: string;
  task_queue?: string;
  timestamp: string; // RFC3339 format

  // Activity-specific fields
  activity_id?: string;
  activity_type?: string;
  activity_input?: any[];
  activity_output?: any;
  spans?: Span[];
  status?: "completed" | "failed";
  duration_ms?: number;
  error?: ErrorDetails;

  // Hook-level governance
  hook_trigger?: true;  // Simple boolean when evaluating per-operation
}
```

**Response Schema:**
```typescript
interface GovernanceResponse {
  verdict: "allow" | "constrain" | "require_approval" | "block" | "halt";
  reason?: string;
  policy_id?: string;
  risk_score?: number;

  // Guardrails
  guardrails_result?: {
    input_type: "activity_input" | "activity_output";
    redacted_input: any;
    validation_passed: boolean;
    reasons?: Array<{type: string; field: string; reason: string}>;
  };

  // HITL
  approval_id?: string;
  approval_expiration_time?: string; // ISO 8601

  // v1.1 fields
  trust_tier?: string;
  alignment_score?: number;
  behavioral_violations?: string[];
  constraints?: any[];
}
```

##### POST /api/v1/governance/approval
**Purpose:** Poll approval status for HITL

**Request Schema:**
```typescript
interface ApprovalRequest {
  workflow_id: string;
  run_id: string;
  activity_id: string;
}
```

**Response Schema:**
```typescript
interface ApprovalResponse {
  verdict: "allow" | "block" | "halt" | "require_approval";
  reason?: string;
  approval_expiration_time?: string; // ISO 8601
  expired?: boolean; // SDK-computed field
}
```

##### GET /api/v1/auth/validate
**Purpose:** Validate API key on SDK initialization

**Headers:**
```
Authorization: Bearer {api_key}
```

**Response:** `200 OK` for valid key, `401/403` for invalid

---

## Data Flow Diagrams

### Workflow Lifecycle Flow

```
┌───────────────┐
│ User starts   │
│ workflow      │
└───────┬───────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│ GovernanceInterceptor.execute_workflow()              │
│                                                       │
│ 1. Call send_governance_event activity               │
│    → WorkflowStarted event                           │
│                                                       │
│ 2. Execute user workflow code                        │
│    - Activities run with ActivityGovernanceInterceptor│
│    - Signals handled with GovernanceInterceptor      │
│                                                       │
│ 3a. Workflow succeeds                                │
│     → Call send_governance_event activity            │
│     → WorkflowCompleted event                        │
│     → Return result                                  │
│                                                       │
│ 3b. Workflow fails                                   │
│     → Extract exception chain                        │
│     → Call send_governance_event activity            │
│     → WorkflowFailed event                           │
│     → Re-raise exception                             │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────┐
│ OpenBox Core          │
│ Evaluates policies    │
│ Returns verdict       │
└───────────────────────┘
```

### Activity Execution Flow with Hook-Level Governance

```
┌───────────────┐
│ Workflow      │
│ calls activity│
└───────┬───────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ ActivityGovernanceInterceptor.execute_activity()              │
│                                                               │
│ 1. Pre-execution checks                                      │
│    - Check pending BLOCK/HALT verdict → fail if present      │
│    - Check pending approval → poll if present                │
│                                                               │
│ 2. Send ActivityStarted event (optional)                     │
│    - Captures activity_input                                 │
│    - Returns verdict + guardrails                            │
│                                                               │
│ 3. Execute activity                                          │
│    - Create OTel span (temporal.workflow_id attribute)       │
│    - Register trace_id → workflow_id mapping                 │
│    - User activity code runs:                                │
│      * HTTP calls → hooks build span_data → governance       │
│        evaluates at started/completed stages                 │
│      * DB queries → hooks build span_data → governance       │
│        evaluates at started/completed stages                 │
│      * File operations → hooks build span_data → governance  │
│      * Functions → @traced evaluates at started/completed    │
│    - Child spans captured automatically                      │
│                                                               │
│ 4. Send ActivityCompleted event                              │
│    - Captures activity_output                                │
│    - Includes all child spans                                │
│    - Returns verdict + guardrails                            │
│                                                               │
│ 5. Return result (or raise exception from hook governance)   │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────┐
│ OpenBox Core          │
│ Evaluates policies    │
│ Returns verdict       │
└───────────────────────┘
```

### HITL Approval Flow

```
┌─────────────────────────────────────────────────────────┐
│ ActivityStarted event sent                              │
│ OpenBox Core returns verdict: "require_approval"        │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│ ActivityInterceptor raises ApplicationError             │
│ - type: "ApprovalPending"                               │
│ - non_retryable: False (Temporal will retry)            │
│ - buffer.pending_approval = True                        │
└────────────┬────────────────────────────────────────────┘
             │
             ▼ (Temporal retry with backoff)
┌─────────────────────────────────────────────────────────┐
│ Activity retries                                        │
│ - Check buffer.pending_approval == True                 │
│ - Poll POST /api/v1/governance/approval                 │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
     ┌───────────────┐
     │ Check response│
     └───────┬───────┘
             │
     ┌───────┴────────────────────┬────────────────┐
     │                            │                │
     ▼                            ▼                ▼
┌────────────┐          ┌──────────────┐   ┌──────────────┐
│verdict:    │          │verdict:      │   │expired: true │
│"allow"     │          │"block"/"halt"│   │              │
└─────┬──────┘          └──────┬───────┘   └──────┬───────┘
      │                        │                   │
      ▼                        ▼                   ▼
┌────────────┐          ┌──────────────┐   ┌──────────────┐
│Clear       │          │Raise non-    │   │Raise non-    │
│pending     │          │retryable     │   │retryable     │
│Proceed     │          │error         │   │error         │
└────────────┘          └──────────────┘   └──────────────┘
```

### Hook-Level Governance Flow

```
┌──────────────────────────────────────────────────────┐
│ Operation detected by hook (HTTP, DB, File, @traced) │
└────────────┬───────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────┐
│ Hook builds span_data dict                           │
│ - hook_type: "http_request|db_query|file_operation" │
│ - stage: "started" or "completed"                    │
│ - Type-specific fields at root level                 │
│   (http_method, http_url, http_status_code, etc.)   │
└────────────┬───────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────┐
│ Hook calls _hook_gov.evaluate_sync/async()           │
│ - Passes span and span_data                          │
└────────────┬───────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────┐
│ hook_governance looks up activity context            │
│ - Uses trace_id to find workflow_id + activity_id    │
│ - Builds full payload with hook_trigger=true         │
└────────────┬───────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────┐
│ POST /api/v1/governance/evaluate                     │
│ - Sends payload with single span                     │
│ - Waits for verdict response                         │
└────────────┬───────────────────────────────────────┘
             │
             ▼
     ┌───────────────┐
     │ Check verdict │
     └───────┬───────┘
             │
     ┌───────┴──────────────────┬──────────────┐
     │                          │              │
     ▼                          ▼              ▼
┌────────────┐      ┌───────────────────┐  ┌────────────┐
│"allow" or  │      │"block", "halt",   │  │No error    │
│"constrain" │      │"require_approval" │  │(fail_open) │
└─────┬──────┘      └─────┬─────────────┘  └─────┬──────┘
      │                    │                      │
      ▼                    ▼                      ▼
┌────────────┐   ┌──────────────────┐   ┌──────────────┐
│Continue    │   │Raise             │   │Log warning,  │
│operation   │   │GovernanceBlocked │   │continue      │
└────────────┘   │Error, abort      │   └──────────────┘
                 │activity          │
                 └──────────────────┘
```

---

## Security Architecture

### Data Privacy

**Design Principle:** Bodies stored separately from OTel spans

```
┌────────────────────────────────────────────────────────┐
│ HTTP Call Made                                         │
└────────────┬───────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ OTel HTTP Instrumentation                              │
│ - Creates span with standard attributes                │
│   {http.method, http.url, http.status_code}            │
│ - NO body data in span attributes                      │
└────────────┬───────────────────────────────────────────┘
             │
             ├─────────────┬────────────────┐
             │             │                │
             ▼             ▼                ▼
┌────────────────┐  ┌──────────────┐  ┌──────────────────┐
│ Span to        │  │ Body to      │  │ Span to          │
│ WorkflowSpan   │  │ span_processor│  │ Fallback OTel    │
│ Processor      │  │ ._body_data  │  │ Exporter (Jaeger)│
│ (governance)   │  │ (private)    │  │ (NO body)        │
└────────────────┘  └──────────────┘  └──────────────────┘
```

**Benefits:**
- Sensitive data never exported to external tracing systems
- Bodies only sent to OpenBox Core (trusted endpoint)
- Ignored URLs (e.g., OpenBox API) completely skip capture

### API Authentication

**API Key Format:** `obx_live_*` or `obx_test_*`

**Validation Flow:**
```
1. SDK initialization
   → Validate key format via regex
   → Call GET /api/v1/auth/validate with Bearer token
   → Raise OpenBoxAuthError if invalid

2. Governance requests
   → Include Authorization: Bearer {api_key} header
   → Server validates on each request
```

### Temporal Sandbox Compliance

**Design Principle:** Strict workflow determinism enforcement

**Prohibited Operations:**
- ❌ Direct HTTP calls (use activities)
- ❌ datetime.now() (use workflow.now())
- ❌ os.stat, os.path.exists (sandbox violation)
- ❌ Module-level imports of httpx, logging, opentelemetry

**Enforcement:**
- Workflow interceptor uses activity for all HTTP
- Lazy imports for non-deterministic libraries
- Public API only exports workflow-safe modules

---

## Scalability & Performance

### Performance Optimizations

1. **Span Buffering** - Batch spans per workflow, send once per activity
2. **Ignored URLs** - Early return to avoid instrumentation overhead
3. **Hook-Level Governance** - Per-operation evaluation (can block early)
4. **Lazy Initialization** - Defer expensive operations until needed
5. **Thread-Safe Locking** - Minimize lock contention with fine-grained locks

### Scalability Limits

| Resource | Limit | Notes |
|----------|-------|-------|
| Concurrent workflows | No SDK limit | Limited by Temporal Server |
| Spans per activity | ~1000 | Practical limit, configurable body size |
| Body size | Configurable | Default: unlimited, set `max_body_size` |
| Governance API timeout | 30s default | Configurable via `api_timeout` |
| Approval polling interval | Temporal retry | Default exponential backoff |

---

## Failure Modes & Resilience

### Failure Scenarios

#### 1. OpenBox Core API Unreachable

**Fail-Open (Default):**
```
1. Activity sends ActivityStarted event
2. HTTP request times out or fails
3. ActivityInterceptor logs warning
4. Returns None (no verdict)
5. Activity proceeds normally
```

**Fail-Closed:**
```
1. Activity sends ActivityStarted event
2. HTTP request times out or fails
3. ActivityInterceptor returns HALT verdict
4. Activity raises ApplicationError (non-retryable)
5. Workflow terminates
```

**Configuration:** `GovernanceConfig.on_api_error = "fail_open" | "fail_closed"`

#### 2. Approval Expired

```
1. Activity requires approval (REQUIRE_APPROVAL verdict)
2. Activity retries, polls approval status
3. approval_expiration_time < current UTC time
4. Response includes expired=true
5. Raise ApplicationError with type="ApprovalExpired" (non-retryable)
6. Workflow terminates
```

#### 3. Stale Verdict from Previous Run

```
1. Workflow run 1 receives BLOCK verdict from signal
2. Workflow restarts (continue-as-new or manual restart)
3. Workflow run 2 starts with different run_id
4. Activity checks verdict.run_id != current run_id
5. Clear stale verdict
6. Activity proceeds normally
```

#### 4. HTTP Body Capture Fails

```
1. HTTP call made via httpx
2. Body capture hook encounters exception
3. Exception caught, logged, ignored
4. Span created WITHOUT body data
5. Event sent with partial telemetry
```

---

## Deployment Architecture

**Recommended Setup:** Run Temporal workers with OpenBox SDK enabled across worker pods. Configure via environment variables: `OPENBOX_URL`, `OPENBOX_API_KEY`, `OPENBOX_GOVERNANCE_TIMEOUT`, `OPENBOX_GOVERNANCE_POLICY`, `TEMPORAL_HOST`, `TEMPORAL_NAMESPACE`. Store API key in Kubernetes secrets.

---

## Monitoring & Observability

**Metrics:** `openbox.governance.requests`, `openbox.governance.verdict.count`, `openbox.approval.pending.duration`, `openbox.span.buffer.size`

**Logs:** Activity interceptor logs governance verdicts and errors; span processor logs buffer events and ignored URLs.

**Traces (Optional):** Export to external systems (Jaeger, Zipkin) via fallback processor. Bodies excluded for privacy.

---

**Document Version:** 1.2
**Last Updated:** 2026-03-16

See `./docs/project-roadmap.md` for future enhancements and planned features.
