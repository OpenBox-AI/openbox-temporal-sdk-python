# Codebase Summary

**Generated:** 2026-03-16
**Repository:** openbox-temporal-sdk-python
**Version:** 1.1.0 (Alpha)
**Total LOC:** 3,700+ (across 11 Python files)

---

## Overview

OpenBox SDK for Temporal Workflows provides governance and observability for Temporal-based applications through workflow/activity interceptors and OpenTelemetry instrumentation. The codebase follows a modular architecture with clear separation between workflow-safe and activity-only code. Hook-level governance enables real-time per-operation policy evaluation.

---

## Project Structure

```
openbox-temporal-sdk-python/
├── openbox/                    # Main SDK package
│   ├── __init__.py            # Public API exports (workflow-safe only)
│   ├── types.py               # Type definitions (workflow-safe)
│   ├── config.py              # Configuration and initialization
│   ├── worker.py              # Worker factory function
│   ├── workflow_interceptor.py # Workflow lifecycle interceptor
│   ├── activity_interceptor.py # Activity lifecycle interceptor
│   ├── activities.py          # Governance event activity
│   ├── span_processor.py      # OTel span buffering and body storage
│   ├── hook_governance.py     # Hook-level governance evaluation
│   ├── otel_setup.py          # HTTP/DB/File instrumentation setup
│   ├── db_governance_hooks.py # Per-library DB governance wrappers
│   └── tracing.py             # @traced decorator for function tracing
├── README.md                  # User-facing documentation
├── pyproject.toml             # Package metadata and dependencies
└── docs/                      # Technical documentation
    ├── project-overview-pdr.md
    ├── codebase-summary.md
    ├── code-standards.md
    └── system-architecture.md
```

---

## Core Components

### 1. Public API (`__init__.py`)

**Purpose:** Export workflow-safe components only
**Key Exports:**
- `create_openbox_worker()` - Recommended factory function
- `initialize()` - Manual SDK initialization
- `GovernanceConfig` - Configuration dataclass
- `Verdict`, `WorkflowEventType` - Enums
- `WorkflowSpanProcessor` - Span buffering
- `GovernanceInterceptor` - Workflow interceptor

**IMPORTANT:** Does NOT export:
- `ActivityGovernanceInterceptor` - Uses OTel (imports `os.stat` via `importlib_metadata`)
- `send_governance_event` - Uses `httpx` (imports `os.stat`)
- `otel_setup` - Uses OTel
- `tracing` - Uses OTel
- `hook_governance` - Uses `httpx`

**Reason:** Temporal sandbox restrictions forbid `os.stat` in workflow context. These must be imported directly when needed in activity context.

**Lines of Code:** 109

---

### 2. Type Definitions (`types.py`)

**Purpose:** Workflow-safe data structures for governance
**Key Types:**

#### Enums
- `WorkflowEventType` - 6 event types (WorkflowStarted, WorkflowCompleted, WorkflowFailed, SignalReceived, ActivityStarted, ActivityCompleted)
- `Verdict` - 5-tier governance response (ALLOW, CONSTRAIN, REQUIRE_APPROVAL, BLOCK, HALT)

#### Dataclasses
- `WorkflowSpanBuffer` - Buffers spans per workflow, stores verdict and pending approval state
- `GuardrailsCheckResult` - Input/output redaction result with validation status
- `GovernanceVerdictResponse` - Parsed API response with verdict, reason, guardrails

#### Custom Exceptions
- `GovernanceBlockedError` - Raised by hook-level governance when operation is blocked

**Key Methods:**
- `Verdict.from_string()` - Parse v1.0/v1.1 verdict strings with backward compat
- `Verdict.priority` - Priority for aggregation (HALT=5, BLOCK=4, REQUIRE_APPROVAL=3, CONSTRAIN=2, ALLOW=1)
- `Verdict.should_stop()` - True if BLOCK or HALT
- `Verdict.requires_approval()` - True if REQUIRE_APPROVAL
- `GuardrailsCheckResult.get_reason_strings()` - Extract failure reasons

**Lines of Code:** 200+

---

### 3. Configuration (`config.py`)

**Purpose:** SDK initialization and configuration management
**Key Components:**

#### Global Config Singleton
- `_GlobalConfig` - Stores API URL, API key, timeout
- `get_global_config()` - Accessor function
- `initialize()` - Validates API key with server via `/api/v1/auth/validate`

#### GovernanceConfig Dataclass
- `skip_workflow_types` - Workflow types to skip
- `skip_activity_types` - Activity types to skip (default: `{"send_governance_event"}`)
- `skip_signals` - Signal names to skip
- `on_api_error` - "fail_open" (default) or "fail_closed"
- `api_timeout` - Timeout in seconds (default: 30.0)
- `send_start_event` - Send WorkflowStarted events (default: True)
- `send_activity_start_event` - Send ActivityStarted events (default: True)
- `hitl_enabled` - Enable approval polling (default: True)
- `skip_hitl_activity_types` - Activity types to skip approval (default: `{"send_governance_event"}`)

#### Exceptions
- `OpenBoxConfigError` - Base configuration error
- `OpenBoxAuthError` - Invalid API key
- `OpenBoxNetworkError` - Network connectivity issues

**Lines of Code:** 320

---

### 4. Worker Factory (`worker.py`)

**Purpose:** Zero-code setup via `create_openbox_worker()` factory
**Function Signature:**
```python
def create_openbox_worker(
    client: Client,
    task_queue: str,
    workflows: Sequence[Type] = (),
    activities: Sequence[Callable] = (),
    openbox_url: Optional[str] = None,
    openbox_api_key: Optional[str] = None,
    governance_timeout: float = 30.0,
    governance_policy: str = "fail_open",
    send_start_event: bool = True,
    send_activity_start_event: bool = True,
    skip_workflow_types: Optional[set] = None,
    skip_activity_types: Optional[set] = None,
    skip_signals: Optional[set] = None,
    hitl_enabled: bool = True,
    instrument_databases: bool = True,
    db_libraries: Optional[set] = None,
    instrument_file_io: bool = False,
    # ... standard Worker parameters
) -> Worker
```

**Setup Flow:**
1. Validate API key with `initialize()`
2. Create `WorkflowSpanProcessor` with ignored URLs
3. Setup OTel instrumentation via `setup_opentelemetry_for_governance()`
4. Configure hook-level governance via `hook_governance.configure()`
5. Create `GovernanceConfig`
6. Create `GovernanceInterceptor` and `ActivityGovernanceInterceptor`
7. Add `send_governance_event` activity to activities list
8. Return fully configured `Worker`

**Lines of Code:** 280+

---

### 5. Workflow Interceptor (`workflow_interceptor.py`)

**Purpose:** Capture workflow lifecycle events (sent via activity for determinism)
**Key Components:**

#### GovernanceInterceptor (Factory)
- Creates `_Inbound` interceptor class per workflow
- Captures API URL, API key, config via closure

#### _Inbound (Interceptor)
- `execute_workflow()` - Sends WorkflowStarted, WorkflowCompleted, WorkflowFailed
- `handle_signal()` - Sends SignalReceived, stores BLOCK/HALT verdict for activity interceptor

**Event Sending:**
- Uses `workflow.execute_activity("send_governance_event", ...)` for all HTTP calls
- Maintains determinism by delegating HTTP to activity
- Uses `workflow.patched()` for version gates

**Error Handling:**
- Catches `GovernanceAPIError` from activity, re-raises as `GovernanceHaltError`
- Extracts nested exception chains for WorkflowFailed events

**Verdict Storage:**
- If SignalReceived returns BLOCK/HALT, stores verdict in span processor
- Activity interceptor checks this before executing activities

**Lines of Code:** 263

---

### 6. Activity Interceptor (`activity_interceptor.py`)

**Purpose:** Capture activity lifecycle events with input/output and spans
**Key Components:**

#### ActivityGovernanceInterceptor (Factory)
- Stores API URL, API key, span processor, config
- Creates `_ActivityInterceptor` per activity

#### _ActivityInterceptor (Interceptor)
- `execute_activity()` - Main interception logic:
  1. Check for pending BLOCK/HALT verdict from signal handler
  2. Check for pending approval and poll status if present
  3. Register workflow buffer if not exists
  4. Send ActivityStarted event (optional)
  5. Check guardrails validation and apply input redaction
  6. Execute activity with OTel span (enables hook-level governance)
  7. Capture child spans (HTTP, DB, file)
  8. Send ActivityCompleted event with output and spans
  9. Apply output redaction if present
  10. Handle REQUIRE_APPROVAL verdict with retry

**Helper Methods:**
- `_send_activity_event()` - POST to `/api/v1/governance/evaluate`
- `_poll_approval_status()` - POST to `/api/v1/governance/approval`
- `_serialize_value()` - Convert args/results to JSON
- `_deep_update_dataclass()` - Apply guardrails redaction to dataclass fields

**Verdict Enforcement:**
- `ALLOW` - Continue normally
- `CONSTRAIN` - Log and continue
- `REQUIRE_APPROVAL` - Raise retryable `ApprovalPending` error, poll on retry
- `BLOCK` - Raise non-retryable `GovernanceStop` error
- `HALT` - Raise non-retryable `GovernanceStop` error (same as BLOCK at activity level)

**Approval Expiration:**
- Polls approval status with `approval_expiration_time` check
- If expired, terminates with non-retryable `ApprovalExpired` error

**Lines of Code:** 754

---

### 7. Governance Event Activity (`activities.py`)

**Purpose:** Execute HTTP calls to OpenBox Core from workflow context
**Activity:** `send_governance_event`

**Function Signature:**
```python
@activity.defn(name="send_governance_event")
async def send_governance_event(input: Dict[str, Any]) -> Optional[Dict[str, Any]]
```

**Input Fields:**
- `api_url` - OpenBox Core URL
- `api_key` - Bearer token
- `payload` - Event data (without timestamp)
- `timeout` - Request timeout
- `on_api_error` - "fail_open" or "fail_closed"

**Behavior:**
- Adds RFC3339 timestamp to payload
- POSTs to `/api/v1/governance/evaluate`
- Parses verdict from response
- For SignalReceived: Returns verdict dict (workflow interceptor stores it)
- For other events with BLOCK/HALT: Raises non-retryable `ApplicationError`
- On API failure with fail_closed: Raises `GovernanceAPIError`

**Lines of Code:** 163

---

### 8. Span Processor (`span_processor.py`)

**Purpose:** Buffer spans per workflow and merge body/header data
**Key Components:**

#### WorkflowSpanProcessor (OTel SpanProcessor)
- Implements OTel `SpanProcessor` interface (`on_start`, `on_end`, `shutdown`, `force_flush`)
- Buffers spans by `workflow_id` (from span attributes)
- Maps `trace_id` → `workflow_id` for child spans (HTTP calls without workflow_id attribute)
- Stores body/header data separately from OTel spans (privacy)

**Data Structures:**
- `_buffers: Dict[str, WorkflowSpanBuffer]` - workflow_id → buffer
- `_trace_to_workflow: Dict[int, str]` - trace_id → workflow_id
- `_trace_to_activity: Dict[int, str]` - trace_id → activity_id
- `_body_data: Dict[int, dict]` - span_id → {request_body, response_body, request_headers, response_headers}
- `_verdicts: Dict[str, dict]` - workflow_id → {verdict, reason, run_id}
- `_activity_contexts: Dict[str, dict]` - activity_id → {workflow_id, activity_id, activity_type}
- `_activity_aborts: Dict[str, str]` - activity_id → abort_reason (for hook-level governance)
- `_halt_requested: Dict[str, bool]` - activity_id → halt_flag

**Key Methods:**
- `register_workflow()` - Register buffer for workflow
- `register_trace()` - Map trace_id to workflow_id + activity_id
- `get_activity_context_by_trace()` - Look up activity context (used by hook_governance)
- `get_activity_abort()` - Check if activity should abort (set by hook verdict)
- `set_activity_abort()` - Set abort flag (called by hook_governance on blocked verdict)
- `get_halt_requested()` - Check if halt requested
- `set_halt_requested()` - Set halt flag
- `store_body()` - Store HTTP body/header data (called from hooks)
- `set_verdict()` / `get_verdict()` - Store/retrieve BLOCK/HALT verdicts from signals
- `on_end()` - Merge body data into span dict and buffer

**Ignored URLs:**
- Configurable list of URL prefixes to skip (e.g., OpenBox Core API)
- Prevents governance event loops

**Lines of Code:** 400+

---

### 9. Hook-Level Governance (`hook_governance.py`)

**Purpose:** Real-time governance evaluation for each operation (HTTP, DB, file, function)
**Key Components:**

#### Configuration
- `configure()` - Set API URL, key, span processor, timeout, error policy
- `is_configured()` - Check if hook governance is active

#### Span Data Builder
- `extract_span_context()` - Extract span_id, trace_id, parent_span_id as hex strings
- Handles NonRecordingSpan, MagicMock, and missing attributes safely

#### Payload Building
- `_build_payload()` - Assemble evaluation payload from activity context + span data
- Looks up activity context by trace_id
- Tags span_data with activity_id
- Sets `hook_trigger=true` (simple boolean)
- Adds RFC3339 timestamp

#### Verdict Handling
- `_handle_verdict()` - Check response verdict and raise GovernanceBlockedError if blocked
- `_send_and_handle()` - Shared response handler for sync/async

#### Activity Abort Tracking
- `_check_activity_abort()` - Check if activity already aborted by prior hook verdict
- `_set_activity_abort()` - Set abort flag for activity (called on governance block)
- `_resolve_activity_ids()` - Map span → (workflow_id, activity_id)

#### Evaluation Functions
- `evaluate_sync()` - Synchronous governance evaluation
  - Checks activity abort status first (short-circuit)
  - Builds payload from activity context + span_data
  - POSTs to `/api/v1/governance/evaluate`
  - Handles BLOCK/HALT/REQUIRE_APPROVAL verdicts
  - Raises GovernanceBlockedError if blocked
- `evaluate_async()` - Async version (same logic)

**Key Characteristics:**
- Module-level configuration (set once by `create_openbox_worker()`)
- Persistent sync/async HTTP clients (lazy init, thread-safe)
- Activity abort tracking prevents double-evaluation
- Fail-open/fail-closed error policies

**Lines of Code:** 375

---

### 10. OpenTelemetry Setup (`otel_setup.py`)

**Purpose:** Instrument HTTP, database, and file I/O libraries
**Main Function:** `setup_opentelemetry_for_governance()`

#### HTTP Instrumentation
**Libraries:**
- `requests` - Hooks for request/response bodies
- `httpx` - Hooks + Client.send patching for body capture
- `urllib3` - Hooks for request/response bodies
- `urllib` - Hook for request body only

**Hooks:**
- `_requests_request_hook()` / `_requests_response_hook()`
- `_httpx_request_hook()` / `_httpx_response_hook()` / async versions
- `_urllib3_request_hook()` / `_urllib3_response_hook()`
- `_urllib_request_hook()`

**Hook-Level Governance Integration:**
- Both started + completed stages build span_data via `_build_http_span_data()`
- Calls `_hook_gov.evaluate_sync()` or `evaluate_async()` with span_data parameter
- Span data includes type-specific fields at root: http_method, http_url, http_status_code, request/response bodies, headers, hook_type

**Body Capture:**
- Only text content types (`text/*`, `application/json`, `application/xml`, etc.)
- Bodies stored via `span_processor.store_body(span_id, request_body=..., response_body=...)`
- Headers captured separately

**Span Data Builders:**
- `_build_http_span_data()` - HTTP span data with hook_type="http_request"

#### Database Instrumentation
**Function:** `setup_database_instrumentation(db_libraries)`
**Supported Libraries:**
- `psycopg2` (PostgreSQL sync)
- `asyncpg` (PostgreSQL async)
- `mysql-connector-python`
- `pymysql`
- `pymongo` (MongoDB)
- `redis`
- `sqlalchemy` (ORM)

**Captured Attributes:**
- `db.system` - Database type
- `db.statement` - SQL query or command
- `db.operation` - Operation type (SELECT, INSERT, GET, etc.)
- `db.name`, `db.user`, `net.peer.name`, `net.peer.port`

#### File I/O Instrumentation
**Function:** `setup_file_io_instrumentation()`
**Implementation:** Patches `builtins.open` with `TracedFile` wrapper
**Operations:**
- `file.open` - Span for open() call
- `file.read`, `file.readline`, `file.readlines` - Read operations
- `file.write`, `file.writelines` - Write operations

**Hook-Level Governance Integration:**
- `_evaluate_governance()` method on TracedFile handles started/completed stages
- Both stages build span_data via `_build_file_span_data()` and call governance evaluate
- Can block file operations (open, read, write) if governance returns BLOCK/HALT

**Attributes:**
- `file.path`, `file.mode`, `file.operation`, `file.bytes`, `file.lines`
- `file.total_bytes_read`, `file.total_bytes_written` (on close)
- `openbox.governance.error` (if governance error occurs)

**Span Data Builders:**
- `_build_file_span_data()` - File span data with hook_type="file_operation"

**Skipped Paths:** `/dev/`, `/proc/`, `/sys/`, `__pycache__`, `.pyc`, `.pyo`, `.so`, `.dylib`

**Lines of Code:** 1,200+

---

### 11. DB Governance Hooks (`db_governance_hooks.py`)

**Purpose:** Per-library database governance wrappers for started/completed stages
**Key Components:**

#### Global Configuration
- `configure(span_processor)` — Store span_processor reference for mark_governed() and span data building

#### Span Data Builder
- `_build_db_span_data()` — Creates span data dict with consistent format:
  - `stage` at root level (started/completed)
  - `span_id`, `trace_id`, `parent_span_id` (hex strings)
  - `hook_type`: "db_query"
  - `end_time`: actual timestamp for completed, None for started
  - `status.code`: "ERROR" if error, "UNSET" otherwise
  - Attributes: db.system, db.operation, db.statement, server info, rowcount
  - Type-specific fields at root: `db_system`, `db_operation`, `db_statement`, `server_address`, `server_port`, `duration_ns`

#### Hook Installation Functions
- `setup_psycopg2_hooks()` — wrapt on `cursor.execute/executemany`
- `setup_asyncpg_hooks()` — wrapt on `Connection.execute/fetch/fetchrow/fetchval`
- `setup_mysql_hooks()` — wrapt on `MySQLCursor.execute`
- `setup_pymysql_hooks()` — wrapt on `Cursor.execute`
- `setup_pymongo_hooks()` — wrapt on `Collection` methods (find, insert_one, etc.)
- `setup_redis_hooks()` — returns `(request_hook, response_hook)` for OTel RedisInstrumentor
- `setup_sqlalchemy_hooks(engine)` — SQLAlchemy `before/after_cursor_execute` + `handle_error` events

#### Shared Helpers
- `_classify_sql(query)` — Extract SQL verb (SELECT, INSERT, etc.)
- `_evaluate_started()` / `_evaluate_started_async()` — Send started governance (can block)
  - Passes `span_data=` parameter with structured span info
- `_evaluate_completed()` / `_evaluate_completed_async()` — Send completed governance
  - Passes `span_data=` parameter with duration and result metadata

#### Ordering
- wrapt hooks **must** be installed BEFORE OTel instrumentors
- Our hook → OTel wrapper → raw DB method

#### C Extension Handling
- Some libraries (psycopg2) have C extension types that `wrapt` cannot patch
- `TypeError` caught gracefully — hooks silently skipped, OTel spans still work

**Lines of Code:** 900+

---

### 12. Function Tracing (`tracing.py`)

**Purpose:** `@traced` decorator for custom function tracing with hook-level governance
**Key Components:**

#### Span Data Builder
- `_build_traced_span_data()` — Creates span data dict with consistent format:
  - `stage` at root level (started/completed)
  - `span_id`, `trace_id`, `parent_span_id` (hex strings)
  - `hook_type`: "function_call"
  - `end_time`: actual timestamp for completed, None for started
  - `status.code`: "ERROR" if error, "UNSET" otherwise
  - Type-specific fields at root: `code_function`, `code_namespace`

#### @traced Decorator
- Supports sync and async functions
- Creates OTel span with function name as span name
- Captures arguments and return values (configurable)
- Handles exceptions with error attributes
- **Sends hook-level governance evaluations** at `started` and `completed` stages (when `hook_governance` is configured)

**Governance Integration:**
- Both started + completed stages pass `span_data=` parameter to evaluate functions
- `started` stage: Function name, module, arguments (if enabled)
- `completed` stage: Result or error info (if enabled)
- Can be blocked at either stage via BLOCK/HALT verdicts
- Zero overhead when `hook_governance` is not configured

**Parameters:**
- `name` - Custom span name (default: function name)
- `capture_args` - Capture positional/keyword args (default: True)
- `capture_result` - Capture return value (default: True)
- `capture_exception` - Capture exception details (default: True)
- `max_arg_length` - Max serialized arg length (default: 2000 chars)

**Attributes:**
- `code.function` - Function name
- `code.namespace` - Module name
- `function.arg.N` - Positional args (JSON serialized)
- `function.kwarg.X` - Keyword args (JSON serialized)
- `function.result` - Return value (JSON serialized)
- `error`, `error.type`, `error.message` - Exception details

#### create_span() Helper
- Manual span creation context manager
- Allows custom attributes and nested spans

**Lines of Code:** 450+

---

## Key Design Patterns

### 1. Temporal Determinism Compliance
- **Workflow interceptor**: No HTTP, no datetime, sends events via activity
- **Activity interceptor**: Direct HTTP, datetime allowed
- **Lazy imports**: httpx, datetime, logging imported only in activity context
- **Version gates**: `workflow.patched()` for safe rollout

### 2. Hook-Level Governance Span Data Pattern
- **Consistent builders**: All hook types (`_build_http_span_data`, `_build_file_span_data`, `_build_db_span_data`, `_build_traced_span_data`) follow same format
- **Stage at root**: `stage` field indicates "started" or "completed"
- **Type-specific fields at root**: All fields for operation type (http_method, db_system, etc.) at root level, NOT in attributes
- **attributes field**: Contains ONLY original OTel attributes (no custom fields injected)
- **hook_type discrimination**: `hook_type` field identifies operation type
- **Timing**: `start_time` (always), `end_time` (completed only), `duration_ns` (if available)
- **Error tracking**: `status.code` = "ERROR" if error, "UNSET" otherwise
- **Single span per evaluation**: No accumulated history
- **Safe span context**: Handles NonRecordingSpan + MagicMock fallback for testing

### 3. Hook_trigger Simplification
- **Before**: Dict with type/stage/data fields
- **After**: Simple boolean `true` in payload
- **Benefit**: Cleaner API, reduces payload size, consistent with all hook types

### 4. Privacy-First Body Capture
- Bodies stored in `WorkflowSpanProcessor._body_data`, NOT in OTel span attributes
- Bodies merged into span dict only when sent to OpenBox Core
- Optional fallback OTel processor receives spans WITHOUT bodies

### 5. Verdict Priority System
- Verdicts have numeric priority (HALT=5, BLOCK=4, REQUIRE_APPROVAL=3, CONSTRAIN=2, ALLOW=1)
- `Verdict.highest_priority()` aggregates multiple verdicts
- Used when multiple policies apply to same event

### 6. Guardrails Deep Redaction
- `_deep_update_dataclass()` recursively updates nested dataclass fields
- Preserves type information while applying redactions
- Supports both dataclass and dict structures

### 7. HITL Approval Polling
- Pending approval stored in `WorkflowSpanBuffer.pending_approval`
- Activity raises retryable error on REQUIRE_APPROVAL verdict
- On retry, polls `/api/v1/governance/approval` endpoint
- Expiration time checked against UTC timestamp
- Cleared on approval/rejection/expiration

### 8. Verdict Staleness Prevention
- Verdicts stored with `run_id` to detect workflow restarts
- Stale verdicts cleared when `run_id` mismatch detected
- Prevents verdicts from previous run affecting new run

### 9. Activity Abort Tracking (Hook Governance)
- Hook verdicts set `_activity_aborts[activity_id]` in span processor
- Subsequent hooks check abort status and short-circuit
- Prevents duplicate evaluations and ensures consistent verdict

---

## Code Statistics

| File | LOC | Purpose |
|------|-----|---------|
| `__init__.py` | 109 | Public API exports |
| `types.py` | 200+ | Type definitions |
| `config.py` | 320 | Configuration |
| `worker.py` | 280+ | Worker factory |
| `workflow_interceptor.py` | 263 | Workflow events |
| `activity_interceptor.py` | 754 | Activity events |
| `activities.py` | 163 | Governance activity |
| `span_processor.py` | 400+ | Span buffering |
| `hook_governance.py` | 375 | Hook-level governance evaluation |
| `otel_setup.py` | 1,200+ | Instrumentation + span data builders |
| `db_governance_hooks.py` | 900+ | DB governance hooks + span data builder |
| `tracing.py` | 450+ | @traced decorator + span data builder |
| **Total** | **~5,000+** | **Core SDK** |

---

## Testing Status

**IMPORTANT:** Comprehensive test suite implemented with 13+ test files.

### Test Files

| Test File | Coverage |
|-----------|----------|
| `test_activities.py` | Governance event activity submission |
| `test_activity_interceptor.py` | Activity-level governance, redaction, approval polling |
| `test_config.py` | SDK initialization, API key validation, URL security |
| `test_db_governance_hooks.py` | DB governance hooks: redis, sqlalchemy, fail policies, schema |
| `test_hook_governance.py` | Hook-level governance: payload building, verdict handling, abort tracking |
| `test_otel_setup.py` | OpenTelemetry instrumentation (HTTP, DB, File I/O) |
| `test_span_processor.py` | Span buffering, body storage, verdict tracking, activity context lookup |
| `test_tracing.py` | @traced decorator for custom function tracing |
| `test_types.py` | Type definitions and verdict conversions |
| `test_worker.py` | Worker factory and setup flow |
| `test_workflow_interceptor.py` | Workflow lifecycle event capture |
| `test_*.py` | Full determinism compliance, error handling, edge cases |

### Test Coverage Areas

- Type conversions and verdict parsing (v1.0/v1.1 compatibility)
- Hook-level governance: payload building, verdict handling, abort tracking
- Span buffering, body storage, and HTTP header capture
- Guardrails input/output redaction (dataclass and dict)
- Configuration validation and API key format checks
- HITL approval polling with expiration handling
- Error policies (fail_open vs fail_closed)
- Database governance hooks (started/completed), file I/O instrumentation
- Temporal determinism compliance

---

## Common Pitfalls

### 1. Module-Level Imports in Workflow Code
**Problem:** Importing `httpx`, `datetime`, or `logging` at module level triggers Temporal sandbox violations
**Solution:** Lazy imports in functions, or import only in activity context

### 2. Forgetting to Add send_governance_event Activity
**Problem:** Workflow interceptor calls activity that doesn't exist
**Solution:** Use `create_openbox_worker()` which adds it automatically, or manually add to activities list

### 3. Body Capture Not Working for httpx
**Problem:** Response body is None even though request succeeded
**Solution:** Ensure `setup_httpx_body_capture()` is called (automatic in `create_openbox_worker()`)

### 4. Stale Verdicts After Workflow Restart
**Problem:** BLOCK verdict from previous run affects new run
**Solution:** SDK clears verdicts with mismatched `run_id` automatically

### 5. Approval Never Expires
**Problem:** `approval_expiration_time` not checked or parsed incorrectly
**Solution:** SDK parses ISO 8601 timestamps and compares against UTC time

### 6. Hook-Level Governance Not Blocking Operations
**Problem:** HTTP request proceeds even though hook governance returned BLOCK
**Solution:** Ensure hook_governance is configured via `create_openbox_worker()` and `hook_trigger=true` is in payload

---

**Document Version:** 1.2
**Last Updated:** 2026-03-16
