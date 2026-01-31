# System Architecture

**Last Updated:** 2026-01-31
**Version:** 1.0.0
**Total LOC:** 3,539 (across 10 Python files)

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

### 3. Instrumentation Layer

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

2. **Custom Hooks** - Capture bodies/headers via `span_processor.store_body()`
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

3. **Client.send Patching** (httpx only) - Reliable body capture
   ```python
   def _patched_async_send(self, request, *args, **kwargs):
       request_body = request.content
       response = await _original_async_send(self, request, *args, **kwargs)
       response_body = response.text
       span_processor.store_body(span_id, request_body=..., response_body=...)
       return response
   ```

**Code Location:** `openbox/otel_setup.py` (lines 1-898)

#### Database Instrumentation

**Supported Databases:**
- PostgreSQL: `psycopg2` (sync), `asyncpg` (async)
- MySQL: `mysql-connector-python`, `pymysql`
- MongoDB: `pymongo`
- Redis: `redis`
- SQLAlchemy: `sqlalchemy` (ORM)

**Instrumentation Strategy:**

1. **OTel Instrumentors** - Create spans with db.* attributes
   ```python
   from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
   Psycopg2Instrumentor().instrument()
   ```

2. **Span Capture** - Automatically buffered by WorkflowSpanProcessor
   ```python
   # Span attributes:
   {
       "db.system": "postgresql",
       "db.statement": "SELECT * FROM users WHERE id = $1",
       "db.operation": "SELECT",
       "db.name": "mydb",
   }
   ```

**Code Location:** `openbox/otel_setup.py` (lines 335-443)

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

2. **Wrap File Operations** - Trace each read/write
   ```python
   class TracedFile:
       def read(self, size=-1):
           with tracer.start_as_current_span("file.read") as span:
               data = self._file.read(size)
               span.set_attribute("file.bytes", len(data))
               return data
   ```

**Skipped Paths:** `/dev/`, `/proc/`, `/sys/`, `__pycache__`, `.pyc`, `.so`

**Code Location:** `openbox/otel_setup.py` (lines 188-332)

---

### 4. Governance Integration Layer

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

### Activity Execution Flow

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
│    - If BLOCK/HALT → raise ApplicationError                  │
│    - If validation_passed=false → raise ApplicationError     │
│    - If REQUIRE_APPROVAL → raise ApprovalPending (retryable) │
│    - If guardrails redaction → apply to input                │
│                                                               │
│ 3. Execute activity                                          │
│    - Create OTel span (temporal.workflow_id attribute)       │
│    - Register trace_id → workflow_id mapping                 │
│    - User activity code runs                                 │
│    - Child spans captured (HTTP, DB, file)                   │
│                                                               │
│ 4. Send ActivityCompleted event                              │
│    - Captures activity_output                                │
│    - Includes all child spans                                │
│    - Returns verdict + guardrails                            │
│    - If BLOCK/HALT → raise ApplicationError                  │
│    - If validation_passed=false → raise ApplicationError     │
│    - If REQUIRE_APPROVAL → raise ApprovalPending (retryable) │
│    - If guardrails redaction → apply to output               │
│                                                               │
│ 5. Return result (or raise exception)                        │
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

### Span Buffering Flow

```
┌──────────────────────────────────────────────────────┐
│ Activity starts                                      │
│ - ActivityInterceptor creates OTel span              │
│ - span_processor.register_trace(trace_id, wf_id)    │
└────────────────┬─────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────┐
│ Activity makes HTTP call (e.g., httpx.post())        │
│                                                      │
│ 1. OTel httpx instrumentation creates child span     │
│    - Shares same trace_id as parent activity span    │
│    - Does NOT have temporal.workflow_id attribute    │
│                                                      │
│ 2. _httpx_request_hook() captures request body      │
│    - span_processor.store_body(span_id, request_body)│
│                                                      │
│ 3. _httpx_response_hook() captures response body    │
│    - span_processor.store_body(span_id, response_body)│
│                                                      │
│ 4. HTTP span ends                                    │
│    - span_processor.on_end(span) called              │
│    - Looks up workflow_id via trace_id mapping       │
│    - Merges body data from _body_data                │
│    - Appends span to workflow buffer                 │
└────────────────┬─────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────┐
│ Activity completes                                   │
│ - ActivityInterceptor retrieves buffer.spans         │
│ - Filters by activity_id                             │
│ - Sends to OpenBox Core in ActivityCompleted event   │
└──────────────────────────────────────────────────────┘
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
3. **Lazy Initialization** - Defer expensive operations until needed
4. **Thread-Safe Locking** - Minimize lock contention with fine-grained locks

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

### Recommended Deployment

```
┌─────────────────────────────────────────────────────────────────┐
│                      Kubernetes Cluster                         │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                   Temporal Worker Pods                    │ │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐          │ │
│  │  │ Worker 1   │  │ Worker 2   │  │ Worker N   │          │ │
│  │  │ (OpenBox   │  │ (OpenBox   │  │ (OpenBox   │          │ │
│  │  │  enabled)  │  │  enabled)  │  │  enabled)  │          │ │
│  │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘          │ │
│  │        │                │                │                 │ │
│  └────────┼────────────────┼────────────────┼─────────────────┘ │
│           │                │                │                   │
│           └────────────────┴────────────────┘                   │
│                            │                                    │
└────────────────────────────┼────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ OpenBox Core    │  │ Temporal Server │  │ External DBs    │
│ (API + Policies)│  │ (gRPC)          │  │ (PostgreSQL,    │
│                 │  │                 │  │  Redis, etc.)   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Configuration Management

**Environment Variables:**
```bash
OPENBOX_URL=https://api.openbox.ai
OPENBOX_API_KEY=obx_live_xyz123
OPENBOX_GOVERNANCE_TIMEOUT=30.0
OPENBOX_GOVERNANCE_POLICY=fail_closed

TEMPORAL_HOST=temporal-frontend.temporal.svc.cluster.local:7233
TEMPORAL_NAMESPACE=default
TEMPORAL_TASK_QUEUE=my-task-queue
```

**Kubernetes Secret:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: openbox-credentials
type: Opaque
data:
  api-key: b2J4X2xpdmVfeHl6MTIz  # base64 encoded
```

---

## Monitoring & Observability

### Metrics (Recommended)

- `openbox.governance.requests.total` - Total governance API calls
- `openbox.governance.requests.errors` - Failed governance API calls
- `openbox.governance.verdict.count{verdict}` - Verdicts by type
- `openbox.approval.pending.duration` - Time spent waiting for approval
- `openbox.span.buffer.size` - Number of spans buffered per workflow

### Logs (Built-in)

```python
# Activity interceptor logs
activity.logger.info("Governance verdict: allow")
activity.logger.warning("Approval expired")
activity.logger.error("API error", exc_info=True)

# Span processor logs
logger.info("Registered workflow buffer")
logger.debug("Stored body for span")
logger.warning("Ignored URL: http://openbox.ai")
```

### Traces (Optional)

Export spans to external tracing system (Jaeger, Zipkin) via fallback processor:

```python
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Create Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger-agent",
    agent_port=6831,
)

# Create span processor with fallback
span_processor = WorkflowSpanProcessor(
    fallback_processor=BatchSpanProcessor(jaeger_exporter),
)
```

**Note:** Bodies are NOT exported to fallback processor (privacy).

---

## Future Enhancements

### Roadmap

**Q2 2026:**
- Approval UI in OpenBox Dashboard
- Constraint enforcement (rate limiting, sandboxing)
- Replay detection and handling

**Q3 2026:**
- Policy DSL for client-side evaluation
- Span sampling for high-volume workflows
- Multi-region OpenBox Core support

**Q4 2026:**
- gRPC support for governance API
- Advanced guardrails (LLM-based redaction)
- Workflow-level approval (not just activity-level)

---

**Document Version:** 1.0
**Last Updated:** 2026-01-28
