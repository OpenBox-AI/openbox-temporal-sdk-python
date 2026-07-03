# System Architecture

**Last Updated:** 2026-07-03
**Version:** 1.3.0

---

## Overview

OpenBox SDK for Temporal Workflows is a governance and observability layer that sits between Temporal workflows and OpenBox Core.

Responsibilities are split across two packages:

- **Base SDK (`openbox_core`)** — HTTP/DB/file/function hook instrumentation, the hook payload shape, hook evaluation, and enforcement. The Temporal worker/plugin builds and owns an `openbox_core` runtime that installs all of this instrumentation.
- **Temporal SDK (this package)** — lifecycle governance (workflow/activity/signal events sent via workflow-safe activities), HITL retry, signal-verdict bridging, and completed-hook stop/halt propagation, coordinated through `TemporalGovernanceState`.

The worker/plugin captures workflow and activity lifecycle events and sends them to OpenBox Core for policy evaluation, while the base runtime it installs governs each HTTP call, database query, file operation, and `@traced` function in real time.

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
│               │  state=                       │  state=              │
│               ▼                               ▼                      │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │        TemporalGovernanceState (governance_state.py)          ││
│  │  ────────────────────────────────────────────────────────────  ││
│  │  Workflow-safe, run-scoped state shared by both interceptors:  ││
│  │  - Signal verdicts (BLOCK/HALT fail the next activity)         ││
│  │  - HITL pending-approval markers (retry polls status)          ││
│  │  - Completed-hook stop bridge (BLOCK skip / HALT terminate)    ││
│  └────────────────────────────────────────────────────────────────┘│
│               │                                                      │
│               ▼  (verdict → Temporal effect)                        │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │      TemporalFrameworkAdapter + core_activity_scope           ││
│  │                        (core_adapter.py)                       ││
│  │  ────────────────────────────────────────────────────────────  ││
│  │  - Maps base verdicts onto Temporal-native effects            ││
│  │  - core_activity_scope binds the shared ActivityContext       ││
│  │    (the only hook-context bridge)                              ││
│  └───────────────────────────────┬────────────────────────────────┘│
└──────────────────────────────────┼──────────────────────────────────┘
                                    │  builds + owns
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│            openbox_core Base Runtime (installed by worker/plugin)   │
│  ────────────────────────────────────────────────────────────────   │
│  Owns HTTP/DB/file/function hook instrumentation, the hook payload   │
│  shape, hook evaluation, and enforcement:                            │
│    HTTP:      httpx, requests, urllib3, urllib                       │
│    Database:  PostgreSQL, MySQL, SQLite, MongoDB, Redis, SQLAlchemy  │
│    File I/O:  open(), read(), write()                                │
│    Functions: @traced (wraps openbox_core governed())               │
│  Resolves per-operation activity context from the ContextStore that │
│  core_activity_scope binds into.                                     │
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

## Plugin Integration

### OpenBoxPlugin (SimplePlugin)

**File:** `openbox/plugin.py`

`OpenBoxPlugin` extends Temporal's `SimplePlugin` base class, providing a drop-in integration for the AI Partner Ecosystem. It composes the SDK's interceptors, the base-SDK runtime, and the governance activity.

```
Worker(client, task_queue, workflows, activities,
       plugins=[OpenBoxPlugin(openbox_url=..., openbox_api_key=...)])

OpenBoxPlugin.__init__():
  → validate_api_key()                        # config.py (also loads Ed25519 signer)
  → TemporalGovernanceState()                 # governance_state.py
  → create_core_runtime(state=...)            # core_adapter.py (base runtime)
  → runtime.install_instrumentation()         # installs HTTP/DB/file/function hooks
  → GovernanceConfig(...)                      # config.py
  → GovernanceInterceptor(state=...)           # workflow_interceptor.py
  → ActivityGovernanceInterceptor(state=..., client=...)  # activity_interceptor.py
  → TracingInterceptor()                       # temporalio W3C trace propagation
  → SimplePlugin.__init__(interceptors, activities, workflow_runner)

OpenBoxPlugin.configure_worker(config):
  → set_temporal_client(config["client"])       # activities.py
  → super().configure_worker(config)            # appends interceptors, activities
```

Both interceptors now take `state=` (a `TemporalGovernanceState`), not a span processor. `create_openbox_worker(...)` follows the same sequence with the same public signature.

**Key design choices:**
- The base runtime is built and owned by the worker/plugin; `install_instrumentation()` runs once for the process lifetime (idempotent)
- Sandbox passthrough for `opentelemetry` via `workflow_runner` callback
- `configure_worker()` captures Temporal client ref for HALT terminate calls
- Plugin name: `"openbox.OpenBoxPlugin"` per Temporal's naming standard
- `db_libraries` / `sqlalchemy_engine` are accepted for compatibility but ignored: the base runtime installs every available DB instrumentor and governs SQLAlchemy via a global Engine listener (covers pre-existing engines)

---

## Component Architecture

### 1. Interceptor Layer

#### GovernanceInterceptor (Workflow-Level)

**Responsibility:** Capture workflow lifecycle events

**Key Characteristics:**
- Workflow-safe (no HTTP, no datetime, no os.stat)
- Events sent via `send_governance_event` activity for determinism
- Records BLOCK/HALT verdicts from SignalReceived in `TemporalGovernanceState` for the activity interceptor

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
   → If verdict is BLOCK/HALT, calls state.set_signal_verdict(wf, run, verdict, reason)
   → Next activity reads state.get_signal_verdict() and fails
```

**Code Location:** `openbox/workflow_interceptor.py`

#### ActivityGovernanceInterceptor (Activity-Level)

**Responsibility:** Capture activity execution with input/output and spans

**Key Characteristics:**
- Activity-only (direct HTTP allowed)
- Captures activity arguments and return values
- Binds the shared `ActivityContext` via `core_activity_scope` so base hooks fire during user code
- Enforces guardrails redaction
- Polls for HITL approval on retry

**Event Flow:**
```
1. Activity starts
   → Check pending signal verdict via state.get_signal_verdict()
   → Check pending approval via state.has_pending_approval() and poll if present
   → Clear any stale within-activity abort flag from the base ContextStore
   → Send ActivityStarted event (optional)
   → Apply input guardrails if present

2. Activity executes
   → Open OTel span, then core_activity_scope binds the shared ActivityContext
   → User activity code runs
   → Base runtime fires HTTP/DB/file/function hooks and governs each operation;
     a BLOCK/HALT/approval is raised as a Temporal-native error by the adapter

3. Activity completes
   → take_completed_stop() + base within-activity abort flag decide whether the
     operation was governed-stopped (HALT → terminate; BLOCK → skip completed event)
   → Send ActivityCompleted event with input/output (unless aborted)
   → Apply output guardrails if present
   → Handle REQUIRE_APPROVAL verdict (retry with polling)

4. Activity retries (if approval pending)
   → Poll /api/v1/governance/approval
   → If approved: clear pending via state.clear_pending_approval(), proceed
   → If rejected: raise non-retryable error
   → If expired: terminate workflow
```

**Code Location:** `openbox/activity_interceptor.py`

---

### 2. Shared Infrastructure Modules

#### Exception Hierarchy (`openbox/errors.py`)

**Responsibility:** Unified, framework-agnostic exception definitions

**Exception Tree:**
```
OpenBoxError (base)
├── OpenBoxConfigError
│   ├── OpenBoxAuthError
│   ├── OpenBoxNetworkError
│   └── OpenBoxInsecureURLError
├── GovernanceBlockedError
├── GovernanceAPIError
├── GovernanceHaltError
├── ApprovalPending
├── ApprovalExpired
└── GovernanceStop
```

**Used By:** All SDK modules for type-safe error handling

**Code Location:** `openbox/errors.py`

#### Governance Client (`openbox/client.py`)

**Responsibility:** Centralized async HTTP client for activity-level events

**Key Characteristics:**
- Persistent `httpx.AsyncClient` (replaces per-request clients)
- Handles both governance evaluation and approval polling
- Shared with `ActivityGovernanceInterceptor` via dependency injection
- Timeout and retry handling built-in

**Code Location:** `openbox/client.py`

#### HITL Module (`openbox/hitl.py`)

**Responsibility:** Encapsulate approval polling and expiration logic

**Key Functions:**
- `poll_approval_status()` - Async polling with expiration check
- `is_approval_expired()` - RFC3339 timestamp comparison

**Used By:** Activity interceptor on REQUIRE_APPROVAL verdict

**Code Location:** `openbox/hitl.py`

#### Verdict Handler (`openbox/verdict_handler.py`)

**Responsibility:** Centralized verdict enforcement (DRY)

**Key Functions:**
- `enforce_verdict()` - Type-safe verdict → exception mapping
  - `ALLOW` → continue
  - `CONSTRAIN` → log warning
  - `REQUIRE_APPROVAL` → raise `ApprovalPending`
  - `BLOCK` / `HALT` → raise `GovernanceStop`

**Used By:** Activity interceptor and workflow interceptor for lifecycle-event verdicts

**Code Location:** `openbox/verdict_handler.py`

---

### 3. Governance State (`governance_state.py`)

#### TemporalGovernanceState

**Responsibility:** Hold the small amount of Temporal semantics that must survive past a base-SDK hook callback — signal verdicts, HITL pending markers, and the completed-hook stop bridge. Workflow-safe, thread-safe, and shared by both interceptors.

The base SDK (`openbox_core`) owns hook context, hook payload building, hook evaluation, and the within-activity abort short-circuit (its `ContextStore`). This object holds only the effects the base runtime cannot express itself. All keys are **run-scoped**: state from a prior run with the same `workflow_id` is ignored and cleared.

**Methods:**
```python
class TemporalGovernanceState:
    # Signal verdicts (workflow interceptor → next activity)
    def set_signal_verdict(self, workflow_id, run_id, verdict, reason=None) -> None
    def get_signal_verdict(self, workflow_id, run_id) -> Optional[(Verdict, reason)]  # clears stale run
    def clear_signal_verdict(self, workflow_id) -> None

    # HITL pending-approval markers
    def mark_pending_approval(self, workflow_id, run_id, activity_id) -> None
    def has_pending_approval(self, workflow_id, run_id, activity_id) -> bool
    def clear_pending_approval(self, workflow_id, run_id, activity_id) -> None

    # Completed-hook stop bridge (adapter → activity interceptor)
    def record_completed_stop(self, workflow_id, run_id, activity_id, verdict, reason=None) -> None
    def take_completed_stop(self, workflow_id, run_id, activity_id) -> Optional[(Verdict, reason)]

    # Cleanup
    def cleanup_run(self, workflow_id, run_id) -> None
```

**Code Location:** `openbox/governance_state.py`

---

### 4. Base Runtime Seam (`core_adapter.py`)

Hook-level governance (per HTTP call, DB query, file operation, and `@traced` function) is owned entirely by the `openbox_core` base runtime. The Temporal SDK builds that runtime, installs it, and maps its verdicts onto Temporal-native effects. This module is NOT sandbox-safe and is never imported from workflow-context code.

#### create_core_runtime + install_instrumentation

**Responsibility:** Build the `openbox_core` runtime that owns all hook instrumentation for a worker/plugin, then install it.

```python
runtime = create_core_runtime(
    api_url=..., api_key=..., state=state,          # TemporalGovernanceState
    timeout_seconds=30.0, on_api_error="fail_open",
    agent_did=None, agent_private_key=None,
    hitl_enabled=True,
    skip_workflow_types=None, skip_activity_types=None, skip_signals=None,
    send_start_event=True, send_activity_start_event=True,
    instrument_databases=True, instrument_file_io=True,
    max_body_size=65536,
)  # -> openbox_core.runtime.OpenBoxRuntime

runtime.install_instrumentation()     # process-lifetime, idempotent
# runtime.uninstall_instrumentation() / runtime.close() for explicit teardown
```

The base runtime installs every available HTTP/DB/file instrumentor and governs SQLAlchemy via a global Engine listener; the caller stores the runtime and owns its lifecycle. The evaluate call never governs itself (the base manager always ignores its own `api_url`).

#### TemporalFrameworkAdapter

**Responsibility:** Map base-SDK governance outcomes onto Temporal-native behavior. Never builds hook payloads or evaluates hook events itself.

**Verdict → Temporal effect mapping:**
- **BLOCK / HALT** (started hook or lifecycle) → non-retryable `temporalio` `ApplicationError`
- **REQUIRE_APPROVAL** → retryable `ApprovalPending` (Temporal's native HITL retry loop) + a pending marker in `TemporalGovernanceState`; if HITL is unavailable for the activity, degrades to a non-retryable block (fail safe)
- **completed-hook BLOCK / HALT** → recorded run-scoped in `TemporalGovernanceState` for the activity interceptor to surface after user code returns

**Methods:** `handle_approval_sync(result, context=None)`, `handle_approval(result)` (async), `on_completed_hook_result(result, context=None)`, `raise_hook_blocked(result)`, `raise_lifecycle_blocked(result)`.

#### core_activity_scope

**Responsibility:** Bind the shared `ActivityContext` around activity execution with a guaranteed try/finally reset.

```python
with core_activity_scope(info, activity_input, trace_id=..., multi_agent_session_id=...):
    result = await self.next.execute_activity(input)
```

This is the **only** hook-context bridge: base instrumentation resolves the activity context from the `ContextStore` this binds into (ambient `ContextVar`, or the trace map for hook code running where `ContextVar`s do not propagate, e.g. executor threads). `get_core_context_store()` and `build_core_activity_context()` support this seam.

**Code Location:** `openbox/core_adapter.py`

#### Hook instrumentation internals

The hook payload shape (per-operation span data for HTTP/DB/file/function), body/header capture, and enforcement now live in **`openbox_core` (base SDK)**. See the base SDK for the exact payload fields and per-library capture strategy — they are no longer defined in this package. Supported surfaces installed by the base runtime include HTTP (`httpx`, `requests`, `urllib3`, `urllib`), databases (PostgreSQL, MySQL, SQLite, MongoDB, Redis, SQLAlchemy), file I/O (`open`/`read`/`write`), and `@traced` functions.

---

### 5. Function Tracing (`tracing.py`)

#### @traced

**Implementation:** `@traced` in `tracing.py` wraps the base SDK's `openbox_core.instrumentation.function.governed` decorator.

**Behavior:**
- With an installed base runtime, a `@traced` function emits started/completed `FUNCTION_CALL` hook events and is governed (can be blocked/halted)
- Without an installed runtime it is a transparent passthrough (zero governance overhead)
- Supports sync and async functions

```python
from openbox.tracing import traced, create_span

@traced
def my_function(arg1, arg2):
    return do_something(arg1, arg2)

@traced(name="custom-name", capture_args=True, capture_result=True)
async def my_async_function(data):
    return await process(data)
```

`capture_exception` and `max_arg_length` are accepted for backward compatibility but no longer tuned here — the base SDK records the completed/error stages and applies its own privacy truncation. `create_span(name, attributes=None)` is a plain OpenTelemetry span helper with no governance. Function-call hook payload building and evaluation live in `openbox_core`.

**Code Location:** `openbox/tracing.py`

---

### 6. Governance Integration Layer

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
│ 1. Pre-execution checks (via TemporalGovernanceState)        │
│    - get_signal_verdict() BLOCK/HALT → fail if present       │
│    - has_pending_approval() → poll approval if present       │
│                                                               │
│ 2. Send ActivityStarted event (optional)                     │
│    - Captures activity_input                                 │
│    - Returns verdict + guardrails                            │
│                                                               │
│ 3. Execute activity                                          │
│    - Open OTel span (temporal.workflow_id attribute)         │
│    - core_activity_scope binds the shared ActivityContext    │
│    - User activity code runs; base runtime governs each op:  │
│      * HTTP calls / DB queries / file ops / @traced funcs    │
│        → base hooks fire at started/completed stages         │
│      * BLOCK/HALT/approval raised as a Temporal-native error │
│        by the adapter (propagates here to fail/retry)        │
│                                                               │
│ 4. Handle completion                                         │
│    - take_completed_stop() + base abort flag decide if the   │
│      operation was governed-stopped (HALT terminate / BLOCK  │
│      skip); otherwise send ActivityCompleted event           │
│    - Captures activity_output, returns verdict + guardrails  │
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
│ - state.mark_pending_approval(wf, run, act)             │
└────────────┬────────────────────────────────────────────┘
             │
             ▼ (Temporal retry with backoff)
┌─────────────────────────────────────────────────────────┐
│ Activity retries                                        │
│ - state.has_pending_approval(wf, run, act) == True      │
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
│clear_      │          │Raise non-    │   │Raise non-    │
│pending_    │          │retryable     │   │retryable     │
│approval()  │          │error         │   │error         │
│Proceed     │          │              │   │              │
└────────────┘          └──────────────┘   └──────────────┘
```

### Hook-Level Governance Flow

Hook detection, payload building, and evaluation are owned by the `openbox_core` base runtime. The Temporal SDK provides the activity-context bridge (`core_activity_scope`) and maps the resulting verdict onto Temporal-native effects (`TemporalFrameworkAdapter`).

```
┌──────────────────────────────────────────────────────┐
│ Operation detected by base runtime hook              │
│ (HTTP, DB, File, @traced) — openbox_core             │
└────────────┬───────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────┐
│ Base runtime resolves activity context               │
│ - From the ContextStore that core_activity_scope     │
│   binds into (ambient ContextVar or trace map)       │
│ - Builds the hook payload + POSTs to Core evaluate   │
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
┌────────────┐   ┌────────────────────────┐  ┌──────────────┐
│Continue    │   │TemporalFrameworkAdapter│  │Log warning,  │
│operation   │   │maps to Temporal effect:│  │continue      │
└────────────┘   │ BLOCK/HALT → non-retry │  └──────────────┘
                 │  ApplicationError      │
                 │ REQUIRE_APPROVAL →     │
                 │  retryable + pending   │
                 │  marker in state       │
                 │ completed BLOCK/HALT → │
                 │  recorded in state     │
                 └────────────────────────┘
```

---

## Security Architecture

### Data Privacy

**Design Principle:** Request/response bodies go only to OpenBox Core, never to external OTel exporters

Body and header capture is owned by the `openbox_core` base runtime (see the base SDK for the exact capture and truncation behavior; body size is bounded by `max_body_size`). Bodies are included only in the payload sent to the trusted OpenBox Core endpoint and are kept out of the standard OTel span attributes exported to external tracing systems.

```
┌────────────────────────────────────────────────────────┐
│ HTTP Call Made                                         │
└────────────┬───────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ openbox_core base runtime hook                         │
│ - Captures request/response bodies + headers           │
│ - OTel span carries only standard attributes           │
│   {http.method, http.url, http.status_code} (NO body)  │
└────────────┬───────────────────────────────────────────┘
             │
             ├──────────────────────────┐
             ▼                          ▼
┌────────────────────────┐   ┌──────────────────────┐
│ Hook payload to        │   │ Span to external OTel │
│ OpenBox Core evaluate  │   │ Exporter (Jaeger etc) │
│ (bodies, trusted)      │   │ (NO body)             │
└────────────────────────┘   └──────────────────────┘
```

**Benefits:**
- Sensitive data never exported to external tracing systems
- Bodies only sent to OpenBox Core (trusted endpoint)
- The OpenBox API URL is skipped so the evaluate call never governs itself

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
   → Include User-Agent: OpenBox-SDK/{version} header
   → Include X-OpenBox-SDK-Version: {version} header
   → Server validates on each request
```

**SDK Version Header:** Core requests include `X-OpenBox-SDK-Version` (and `User-Agent: OpenBox-SDK/{version}`) with the SDK version. The version is a **static `__version__` in `openbox/__init__.py`** — it is intentionally NOT read via `importlib.metadata`, because a metadata lookup opens a file and, with file instrumentation active, would re-enter the file hook. The header is assembled in `request_signing.py`, which imports the static `__version__`.

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

**Document Version:** 1.3
**Last Updated:** 2026-03-23

See `./docs/project-roadmap.md` for future enhancements and planned features.
