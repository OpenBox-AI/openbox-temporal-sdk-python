# Codebase Summary

**Generated:** 2026-03-23
**Repository:** openbox-temporal-sdk-python
**Version:** 1.1.2 (Alpha)
**Total LOC:** ~3,800 (across 18 Python files)

---

## Overview

OpenBox SDK for Temporal Workflows provides governance and observability for Temporal-based applications. It layers on top of the OpenBox base SDK (`openbox_core`): the base SDK owns all per-operation hook instrumentation (HTTP, database, file, and function calls), the hook payload shape, evaluation, and enforcement, while this package owns Temporal lifecycle governance — workflow/activity/signal events sent through workflow-safe activities, human-in-the-loop (HITL) approval via Temporal's retry loop, signal-verdict bridging, and completed-hook stop/halt propagation.

The worker factory and plugin build and own an `openbox_core` runtime and call `install_instrumentation()` once per process; the interceptors then translate base-SDK verdicts into Temporal-native effects. The codebase keeps a clear separation between workflow-safe code (importable inside the Temporal sandbox) and activity-only code (which imports OpenTelemetry / the base runtime / `httpx`).

---

## Project Structure

```
openbox-temporal-sdk-python/
├── openbox/                    # Main SDK package
│   ├── __init__.py            # Public API exports (workflow-safe only)
│   ├── types.py               # Type definitions (shims over openbox_core contracts)
│   ├── config.py              # Configuration and initialization
│   ├── worker.py              # Worker factory function
│   ├── plugin.py              # OpenBoxPlugin(SimplePlugin) for Temporal AI Partner Ecosystem
│   ├── errors.py              # Unified exception hierarchy
│   ├── client.py              # GovernanceClient (lifecycle-event HTTP client)
│   ├── hitl.py                # HITL approval handling
│   ├── verdict_handler.py      # Centralized verdict enforcement
│   ├── governance_state.py    # TemporalGovernanceState — run-scoped effects shared by interceptors
│   ├── core_adapter.py        # Base-SDK runtime builder + Temporal FrameworkAdapter + ActivityContext binding
│   ├── workflow_interceptor.py # Workflow lifecycle interceptor
│   ├── activity_interceptor.py # Activity lifecycle interceptor
│   ├── activities.py          # Governance event activity
│   ├── request_signing.py     # AIP DID + Ed25519 signed-request construction (shim over openbox_core.identity)
│   ├── multi_agent.py         # Multi-agent handoff events + session-context propagation
│   └── tracing.py             # @traced decorator (wraps openbox_core governed()) + create_span
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
- `OpenBoxPlugin` - Temporal plugin (when `temporalio >= 1.23.0`)
- `initialize()`, `get_global_config()` - SDK initialization / config accessor
- `GovernanceConfig` - Configuration dataclass
- `Verdict`, `WorkflowEventType` - Enums (from `openbox_core.contracts`)
- `GovernanceVerdictResponse`, `GuardrailsCheckResult` - Result types
- `GovernanceInterceptor` - Workflow interceptor
- `enforce_verdict`, `VerdictEnforcementResult` - Verdict enforcement
- HITL helpers: `handle_approval_response`, `raise_approval_pending`, `should_skip_hitl`
- `GovernanceClient` - Lifecycle-event HTTP client (httpx imported lazily)
- `emit_handoff` - Multi-agent handoff event
- The unified error hierarchy (`OpenBoxError` and subclasses)

**IMPORTANT:** Does NOT export (import directly in activity context):
- `ActivityGovernanceInterceptor` - Uses OpenTelemetry (imports `os.stat` via `importlib_metadata`). Import from `openbox.activity_interceptor`.
- `send_governance_event` (and `build_governance_activities`) - Uses `httpx` (imports `os.stat`). Import from `openbox.activities`.
- `traced` / `create_span` - Wrap OpenTelemetry / the base `governed()` decorator. Import from `openbox.tracing`.
- `create_core_runtime`, `TemporalFrameworkAdapter`, `core_activity_scope`, `get_core_context_store` - Build/bind the base runtime and activity context; import from `openbox.core_adapter`.

**Reason:** Temporal sandbox restrictions forbid `os.stat` in workflow context. Anything that pulls in the base runtime, OpenTelemetry, or `httpx` must be imported directly when needed in activity context. (`__version__` is a static string on purpose — a metadata lookup would open a file, which re-enters file instrumentation.)

**Lines of Code:** 177

---

### 2. Type Definitions (`types.py`)

**Purpose:** Workflow-safe governance data structures — thin shims over the shared `openbox_core.contracts`, so every OpenBox framework SDK consumes the same shapes. The base contracts are pure (no network, crypto, OTel, or wall-clock at import), so this module stays sandbox-safe.

**Key Types:**

#### Enums (re-exported from `openbox_core.contracts`)
- `WorkflowEventType` - event types (base name `EventType`): WorkflowStarted, WorkflowCompleted, WorkflowFailed, SignalReceived, ActivityStarted, ActivityCompleted
- `Verdict` - 5-tier governance response (ALLOW, CONSTRAIN, REQUIRE_APPROVAL, BLOCK, HALT)

#### Result Types
- `GuardrailsCheckResult` - Input/output redaction result with validation status (base name `GuardrailsResult`)
- `GovernanceVerdictResponse` - Parsed API response; a thin subclass of the shared `EvaluationResult` (adds `fallback_used`, `diagnostics`, `raw` on top of the historical Temporal surface)

#### Custom Exceptions
- `GovernanceBlockedError` - Re-exported from `errors.py` for backward compatibility (defined in the exception hierarchy, not here)

**Key Methods (provided by the shared contracts):**
- `Verdict.from_string()` - Parse v1.0/v1.1 verdict strings with backward compat
- `Verdict.priority` - Priority for aggregation (HALT=5, BLOCK=4, REQUIRE_APPROVAL=3, CONSTRAIN=2, ALLOW=1)
- `Verdict.should_stop()` - True if BLOCK or HALT
- `Verdict.requires_approval()` - True if REQUIRE_APPROVAL
- `GovernanceVerdictResponse.from_dict()` - Parse a governance response via the shared base-SDK parser
- `GuardrailsCheckResult.get_reason_strings()` - Extract failure reasons
- `rfc3339_now()` - Current UTC time in RFC3339 format

**Lines of Code:** 91

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
- `hitl_poll_interval_ms` - Approval polling interval in milliseconds (default: 5000)
- `skip_hitl_activity_types` - Activity types to skip approval (default: `{"send_governance_event"}`)
- `session_id` - Session identifier for governance context (optional)
- `agent_name` - Agent name for governance context (optional)
- `tool_type_map` - Mapping of tool names to types for governance (default: empty dict)

#### Exceptions
- `OpenBoxConfigError` - Base configuration error
- `OpenBoxAuthError` - Invalid API key
- `OpenBoxNetworkError` - Network connectivity issues

**Lines of Code:** 320

---

### 4. Exception Hierarchy (`errors.py`)

**Purpose:** Unified exception hierarchy for SDK errors (framework-agnostic)
**Key Exceptions:**
- `OpenBoxError(Exception)` - Base exception for all SDK errors
- `OpenBoxConfigError(OpenBoxError)` - Configuration-related errors
- `OpenBoxAuthError(OpenBoxConfigError)` - Invalid API key
- `OpenBoxNetworkError(OpenBoxConfigError)` - Network connectivity issues
- `OpenBoxInsecureURLError(OpenBoxConfigError)` - Insecure URL (non-HTTPS)
- `GovernanceBlockedError(OpenBoxError)` - Operation blocked by hook-level governance
- `GovernanceAPIError(OpenBoxError)` - API failure (from activities)
- `GovernanceHaltError(OpenBoxError)` - Workflow-level halt requested
- `ApprovalPending(OpenBoxError)` - Retryable error pending approval
- `ApprovalExpired(OpenBoxError)` - Approval has expired
- `GovernanceStop(OpenBoxError)` - Generic governance stop

**Lines of Code:** 160+

---

### 5. Governance Client (`client.py`)

**Purpose:** Centralized HTTP client for activity-level governance events
**Key Components:**
- `GovernanceClient` - Persistent async HTTP client wrapping httpx.AsyncClient
- `_send_event()` - POST event to `/api/v1/governance/evaluate`
- `_poll_approval()` - POST to `/api/v1/governance/approval`

**Characteristics:**
- Single persistent client per activity lifecycle (no per-request creation)
- Lazy initialization (created on first use)
- Thread-safe client acquisition via asyncio.Lock
- Timeout and retry handling
- Shared with activity interceptor via dependency injection

**Lines of Code:** 200+

---

### 6. HITL Module (`hitl.py`)

**Purpose:** Encapsulate approval polling and expiration logic
**Key Components:**
- `poll_approval_status()` - Async function for polling approval with expiration check
- `is_approval_expired()` - Check if approval_expiration_time has passed

**Integration Points:**
- Called from activity interceptor when REQUIRE_APPROVAL verdict received
- Handles RFC3339 timestamp parsing and UTC comparison
- Raises `ApprovalExpired` if poll result indicates expiration

**Lines of Code:** 80+

---

### 7. Verdict Handler (`verdict_handler.py`)

**Purpose:** Centralized verdict enforcement for the activity interceptor (DRY up duplicate logic). NOT sandbox-safe (uses logging at module level).
**Key Functions:**
- `enforce_verdict(response, context)` - Enforce a governance verdict; returns a `VerdictEnforcementResult`. Priority order: HALT > BLOCK > guardrails > REQUIRE_APPROVAL > CONSTRAIN > ALLOW.
  - `HALT` - Raise `GovernanceHaltError` (workflow termination)
  - `BLOCK` - Raise `GovernanceBlockedError` (non-retryable activity failure)
  - guardrails failure - Raise `GuardrailsValidationError`
  - `REQUIRE_APPROVAL` - Return `VerdictEnforcementResult(requires_hitl=True)`; caller drives the HITL flow
  - `CONSTRAIN` - Log the constraint, allow execution
  - `ALLOW` - No action
- `VerdictEnforcementResult` - Small result object (`requires_hitl`, `blocked`) telling the caller what to do next.

**Used By:**
- `activity_interceptor.py` - After receiving a governance response
- `workflow_interceptor.py` - For signal verdicts (handled inline in the sandbox)

**Lines of Code:** 103

---

### 8. Worker Factory (`worker.py`)

**Purpose:** Zero-code setup via `create_openbox_worker()` factory
**Function Signature (keyword-only after `task_queue`):**
```python
def create_openbox_worker(
    client: Client,
    task_queue: str,
    *,
    workflows: Sequence[Type] = (),
    activities: Sequence[Callable] = (),
    openbox_url: str,
    openbox_api_key: str,
    agent_did: Optional[str] = None,
    agent_private_key: Optional[str] = None,
    governance_timeout: float = 30.0,
    governance_policy: str = "fail_open",
    send_start_event: bool = True,
    send_activity_start_event: bool = True,
    skip_workflow_types: Optional[set] = None,
    skip_activity_types: Optional[set] = None,
    skip_signals: Optional[set] = None,
    hitl_enabled: bool = True,
    instrument_databases: bool = True,
    db_libraries: Optional[set] = None,      # accepted for compat; ignored
    sqlalchemy_engine: Optional[Any] = None, # accepted for compat; ignored
    instrument_file_io: bool = True,
    enable_trace_propagation: bool = True,
    # ... standard Worker parameters
) -> Worker
```

`openbox_url` and `openbox_api_key` are required. `agent_did` + `agent_private_key` (both-or-neither) enable AIP DID + Ed25519 request signing. `db_libraries` and `sqlalchemy_engine` are still accepted but ignored — the base runtime installs every available DB instrumentor and governs SQLAlchemy via a global Engine event listener (covering pre-existing engines).

**Setup Flow:**
1. Store the Temporal client (for HALT terminate calls), then validate the API key + URL security and load the Ed25519 signer via `initialize()`.
2. Create a `TemporalGovernanceState()` — signal verdicts, HITL pending markers, and the completed-hook stop bridge, shared by both interceptors.
3. Build and own the base-SDK runtime via `create_core_runtime(...)`, then call `runtime.install_instrumentation()` (process-lifetime, idempotent). The runtime installs all hook instrumentation (HTTP/DB/file/function) and owns hook payload building, evaluation, and enforcement.
4. Build a `GovernanceConfig` for the lifecycle-event path.
5. Create `GovernanceInterceptor(..., state=..., config=...)` and `ActivityGovernanceInterceptor(..., state=..., config=..., client=...)` — both take a `TemporalGovernanceState`.
6. Build the governance activities (`send_governance_event`, with credentials captured on the instance) and, if `enable_trace_propagation`, append Temporal's built-in `TracingInterceptor` for W3C trace propagation.
7. Return a fully configured `Worker`.

**Lines of Code:** 334

---

### 9. Workflow Interceptor (`workflow_interceptor.py`)

**Purpose:** Capture workflow lifecycle events (sent via activity for determinism)
**Key Components:**

**Constructor:** `GovernanceInterceptor(api_url="", api_key="", state=None, config=None)` — takes a `TemporalGovernanceState` (`state=`).

#### GovernanceInterceptor (Factory)
- Creates `_Inbound` interceptor class per workflow
- Captures API URL, API key, state, config via closure

#### _Inbound (Interceptor)
- `execute_workflow()` - Sends WorkflowStarted, WorkflowCompleted, WorkflowFailed
- `handle_signal()` - Sends SignalReceived; on BLOCK/HALT records the verdict via `state.set_signal_verdict(...)` for the activity interceptor

**Event Sending:**
- Uses `workflow.execute_activity("send_governance_event", ...)` for all HTTP calls
- Maintains determinism by delegating HTTP to activity
- Uses `workflow.patched()` for version gates

**Error Handling:**
- Catches `GovernanceAPIError` from activity, re-raises as `GovernanceHaltError`
- Extracts nested exception chains for WorkflowFailed events

**Verdict Storage:**
- If SignalReceived returns BLOCK/HALT, stores the verdict in `TemporalGovernanceState` (run-scoped)
- Activity interceptor reads it before executing activities and enforces (HALT -> terminate, BLOCK -> non-retryable `ApplicationError` type `"GovernanceBlock"`)

**Lines of Code:** 396

---

### 10. Activity Interceptor (`activity_interceptor.py`)

**Purpose:** Capture activity lifecycle events with input/output; enforce Temporal effects of governance
**Constructor:** `ActivityGovernanceInterceptor(api_url, api_key, state, config=None, client=None)` — takes a `TemporalGovernanceState` (`state=`) and an optional shared `GovernanceClient`.

**Key Components:**

#### ActivityGovernanceInterceptor (Factory)
- Stores API URL, API key, state, config, client
- Creates `_ActivityInterceptor` per activity

#### _ActivityInterceptor (Interceptor)
- `execute_activity()` - Main interception logic:
  1. Check for a pending signal BLOCK/HALT verdict via `state.get_signal_verdict(...)`
  2. Check for a pending approval (`state.has_pending_approval(...)`) and poll status if present
  3. Send ActivityStarted event (optional)
  4. Check guardrails validation and apply input redaction
  5. Bind the shared base `ActivityContext` via `core_activity_scope(...)` and execute the activity (base hook instrumentation governs HTTP/DB/file/function calls)
  6. After user code returns, resolve completed-hook stops and the base within-activity abort flag
  7. Send ActivityCompleted event (unless the operation was governed-stopped)
  8. Apply output redaction if present
  9. Handle REQUIRE_APPROVAL verdict with retry

**Completed-hook / abort handling:**
- `state.take_completed_stop(...)` returns and clears a run-scoped BLOCK/HALT recorded by the adapter during a completed hook
- `get_core_context_store().is_activity_aborted(wf, act)` reports a started-hook BLOCK the user code swallowed; the flag is cleared with `clear_activity_aborted(...)`
- Completed HALT reaches Temporal's terminate path; completed BLOCK skips the ActivityCompleted event

**Verdict Enforcement:**
- `ALLOW` - Continue normally
- `CONSTRAIN` - Log and continue
- `REQUIRE_APPROVAL` - Mark a pending approval in state and raise retryable `ApprovalPending`; poll on retry
- `BLOCK` - Raise non-retryable `ApplicationError` (type `"GovernanceBlock"`)
- `HALT` - Terminate the workflow (via `_terminate_workflow_for_halt`)

**Approval Expiration:**
- Polls approval status with `approval_expiration_time` check
- If expired, fails non-retryably

**Lines of Code:** 616

---

### 11. Governance Event Activity (`activities.py`)

**Purpose:** Execute HTTP calls to OpenBox Core from workflow context
**Activity:** `send_governance_event` (method of `GovernanceActivities` class)

**Class Signature:**
```python
class GovernanceActivities:
    def __init__(self, api_url: str, api_key: str): ...

    @activity.defn(name="send_governance_event")
    async def send_governance_event(
        self, input: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]: ...


def build_governance_activities(api_url: str, api_key: str) -> GovernanceActivities
```

The plugin / `create_openbox_worker()` factory build one instance and register
its bound `send_governance_event` method with the Temporal Worker.
Credentials live on `self` — they do **not** flow through activity inputs,
so the API key is never written to workflow history.

**Input Fields (activity input dict):**
- `payload` - Event data (without timestamp)
- `timeout` - Request timeout
- `on_api_error` - "fail_open" or "fail_closed"

> `api_url` / `api_key` are deliberately absent from the input dict to avoid
> leaking credentials into workflow history. They are held on the activity
> instance itself.

**Behavior:**
- Adds RFC3339 timestamp to payload
- POSTs to `{self._api_url}/api/v1/governance/evaluate`
- Parses verdict from response
- For SignalReceived: Returns verdict dict (workflow interceptor stores it)
- For other events with BLOCK/HALT: Raises non-retryable `ApplicationError`
  with `type="GovernanceBlock"` / `"GovernanceHalt"`
- On API failure with fail_closed: Raises `GovernanceAPIError`

A module-level `send_governance_event(input)` helper remains for direct
callers (tests, scripts) that already hold credentials; it instantiates
`GovernanceActivities` internally and delegates.

---

### 12. Governance State (`governance_state.py`)

**Purpose:** Hold the small amount of Temporal governance state that must survive *past* a base-SDK hook callback. The base SDK owns hook context, payload building, evaluation, and within-activity abort short-circuit (its `ContextStore`); `TemporalGovernanceState` holds only the Temporal effects the base runtime cannot express itself. In-memory, thread-safe (activities run on worker threads), and shared by both interceptors. All keys are **run-scoped**: state from a prior run with the same `workflow_id` is ignored and cleared.

**Class:** `TemporalGovernanceState`

**What it tracks:**
- **Signal verdicts** — a SignalReceived BLOCK/HALT must fail the *next* activity in the same run (workflow interceptor records; activity interceptor enforces).
- **HITL pending-approval markers** — a REQUIRE_APPROVAL raises a retryable error; the next attempt must POLL approval status instead of re-evaluating.
- **Completed-hook stop bridge** — a completed-hook BLOCK/HALT resolved by the base runtime is recorded here (keyed by workflow/run/activity) so the activity interceptor can skip a duplicate completed event (BLOCK) or reach the terminate path (HALT) after user code returns, then clear it.

**Public Methods:**
- `set_signal_verdict(workflow_id, run_id, verdict, reason=None)`
- `get_signal_verdict(workflow_id, run_id) -> Optional[(Verdict, reason)]` (clears a stale prior-run entry)
- `clear_signal_verdict(workflow_id)`
- `mark_pending_approval(workflow_id, run_id, activity_id)`
- `has_pending_approval(workflow_id, run_id, activity_id) -> bool`
- `clear_pending_approval(workflow_id, run_id, activity_id)`
- `record_completed_stop(workflow_id, run_id, activity_id, verdict, reason=None)`
- `take_completed_stop(workflow_id, run_id, activity_id) -> Optional[(Verdict, reason)]` (returns AND clears)
- `cleanup_run(workflow_id, run_id)` (drop all state for a finished run)

**Lines of Code:** 126

---

### 13. Core Adapter (`core_adapter.py`)

**Purpose:** The Temporal side of the `openbox_core` adapter seam. Builds the base-SDK runtime that owns all hook instrumentation, maps base verdicts onto Temporal-native effects, and binds the shared `ActivityContext` around activity execution. **NOT sandbox-safe** — imports `temporalio.exceptions` and the base runtime modules; must never be imported on the workflow sandbox path (guarded by `tests/test_workflow_sandbox_import_safety.py`).

**Public Names:** `create_core_runtime`, `TemporalFrameworkAdapter`, `core_activity_scope`, `get_core_context_store`, `build_core_activity_context`

#### `create_core_runtime(...)`
```python
def create_core_runtime(
    *,
    api_url: str,
    api_key: str,
    state: TemporalGovernanceState,
    timeout_seconds: float = 30.0,
    on_api_error: str = "fail_open",
    agent_did: Optional[str] = None,
    agent_private_key: Optional[str] = None,
    hitl_enabled: bool = True,
    skip_hitl_activity_types: Optional[set] = None,
    skip_workflow_types: Optional[set] = None,
    skip_activity_types: Optional[set] = None,
    skip_signals: Optional[set] = None,
    send_start_event: bool = True,
    send_activity_start_event: bool = True,
    instrument_databases: bool = True,
    instrument_file_io: bool = True,
    max_body_size: int = 65536,
) -> OpenBoxRuntime
```
Builds and returns an `openbox_core.runtime.OpenBoxRuntime` wired with a `TemporalFrameworkAdapter`. The caller (worker/plugin init only, never workflow sandbox) stores the runtime and calls `runtime.install_instrumentation()`. The base InstrumentationManager always ignores `config.api_url`, so the evaluate call never governs itself.

#### `TemporalFrameworkAdapter(state, *, hitl_enabled=True, skip_hitl_activity_types=None, context_store=None)`
Maps base-SDK governance outcomes onto Temporal behavior:
- BLOCK/HALT (started hook or lifecycle) -> non-retryable `temporalio` `ApplicationError`.
- REQUIRE_APPROVAL -> retryable `ApprovalPending` + a pending marker in `TemporalGovernanceState` (so the retry attempt polls instead of re-evaluating). Where HITL is unavailable for the activity, degrades to a non-retryable block (fail safe).
- Completed-hook BLOCK/HALT -> recorded run-scoped in `TemporalGovernanceState` for the activity interceptor to surface after user code returns.

Methods: `handle_approval_sync(result, context=None)`, `handle_approval(result)` (async), `on_completed_hook_result(result, context=None)`, `raise_hook_blocked(result)`, `raise_lifecycle_blocked(result)`.

#### `core_activity_scope(info, activity_input=None, *, trace_id=None, multi_agent_session_id=None)`
Context manager binding the shared `ActivityContext` around activity execution with a guaranteed try/finally reset. This is the only hook-context bridge: base instrumentation resolves the activity context from the process-wide `ContextStore` this binds into (ambient ContextVar, or the trace map for hook code running where ContextVars do not propagate). `build_core_activity_context(info, ...)` constructs the context; `get_core_context_store()` returns the process-wide store.

**Lines of Code:** 277

---

### 14. Hook Instrumentation (owned by `openbox_core`)

HTTP, database, file, and function-call hook instrumentation — plus the hook payload shape, evaluation, and enforcement — now live entirely in the OpenBox base SDK (`openbox_core`). This package no longer instruments any libraries directly; the worker/plugin build an `openbox_core` runtime and call `runtime.install_instrumentation()` once per process. For the payload shape, body-capture privacy rules, per-library DB coverage, and evaluation internals, see the base SDK. This package only bridges those hooks to Temporal effects (via the Core Adapter and `TemporalGovernanceState`).

---

### 15. Function Tracing (`tracing.py`)

**Purpose:** `@traced` decorator for function-call governance — a thin compatibility wrapper over the base SDK. Holds no hook logic of its own.
**Key Components:**

#### @traced Decorator
- Delegates to `openbox_core.instrumentation.function.governed`.
- When a worker/plugin has installed the base runtime, a decorated function emits started/completed FUNCTION_CALL hook events and is subject to governance (can be blocked via BLOCK/HALT).
- Without an installed runtime it is a transparent passthrough.
- Supports sync and async functions.

**Parameters:**
- `name` - Custom span name (default: function name)
- `capture_args` - Capture positional/keyword args (default: True)
- `capture_result` - Capture return value (default: True)
- `capture_exception` - Accepted for backward compatibility; no longer tuned here (the base SDK records completed/error stages)
- `max_arg_length` - Accepted for backward compatibility; the base SDK applies its own privacy truncation

Governance payload building, evaluation, and enforcement live entirely in `openbox_core`.

#### create_span() Helper
- Plain OpenTelemetry span context manager (no governance)
- Allows custom attributes and nested spans

**Lines of Code:** 60

---

## Key Design Patterns

### 1. Temporal Determinism Compliance
- **Workflow interceptor**: No HTTP, no datetime, sends events via activity
- **Activity interceptor**: Direct HTTP, datetime allowed
- **Lazy imports**: httpx, datetime, logging imported only in activity context
- **Version gates**: `workflow.patched()` for safe rollout

### 2. Base-SDK-owned Hook Governance
- HTTP/DB/file/function hook instrumentation, the hook payload shape, evaluation, and enforcement all live in `openbox_core`.
- The worker/plugin build an `openbox_core` runtime and call `runtime.install_instrumentation()` once per process (idempotent, process-lifetime).
- `TemporalFrameworkAdapter` is the verdict-mapping seam: it translates base-SDK outcomes into Temporal-native effects (non-retryable `ApplicationError` for BLOCK/HALT, retryable `ApprovalPending` for REQUIRE_APPROVAL, run-scoped records for completed-hook stops).
- `core_activity_scope(...)` binds the shared `ActivityContext` into a process-wide `ContextStore` so base hook code can resolve which activity it is running inside.

### 3. Verdict Priority System
- Verdicts have numeric priority (HALT=5, BLOCK=4, REQUIRE_APPROVAL=3, CONSTRAIN=2, ALLOW=1) via `Verdict.priority`.
- `enforce_verdict()` applies the order HALT > BLOCK > guardrails > REQUIRE_APPROVAL > CONSTRAIN > ALLOW when a single response could imply several actions.

### 4. Guardrails Deep Redaction
- `_deep_update_dataclass()` (in `activity_interceptor.py`) recursively updates nested dataclass fields
- Preserves type information while applying redactions
- Supports both dataclass and dict structures

### 5. HITL Approval Polling
- A pending approval is tracked in `TemporalGovernanceState` (`mark_pending_approval` / `has_pending_approval`), keyed by (workflow_id, run_id, activity_id).
- On REQUIRE_APPROVAL the activity raises a retryable `ApprovalPending` error.
- The retry attempt sees the marker and polls approval status (instead of re-evaluating), checks the expiration time against UTC, and clears the marker on approval/rejection/expiration.

### 6. Verdict Staleness Prevention
- Signal verdicts are stored in `TemporalGovernanceState` with `run_id` (run-scoped).
- A verdict left by a prior run with the same `workflow_id` is ignored and cleared on read.
- Prevents a verdict from a previous run affecting a new run.

### 7. Activity Abort Tracking (base ContextStore)
- A started-hook BLOCK the user code swallowed is surfaced via the base `ContextStore`: the activity interceptor checks `get_core_context_store().is_activity_aborted(workflow_id, activity_id)` after user code returns and clears it.
- A completed-hook BLOCK/HALT is bridged run-scoped through `TemporalGovernanceState.take_completed_stop(...)`.
- Either signal means the operation was governed-stopped: BLOCK skips the ActivityCompleted event; HALT reaches Temporal's terminate path.

---

## Code Statistics

| File | LOC | Purpose |
|------|-----|---------|
| `__init__.py` | 177 | Public API exports |
| `types.py` | 91 | Type definitions (shims over `openbox_core.contracts`) |
| `config.py` | 478 | Configuration |
| `worker.py` | 334 | Worker factory |
| `errors.py` | 257 | Exception hierarchy |
| `client.py` | 213 | Lifecycle-event HTTP client |
| `hitl.py` | 137 | Approval polling helpers |
| `verdict_handler.py` | 103 | Verdict enforcement |
| `governance_state.py` | 126 | Run-scoped Temporal governance state |
| `core_adapter.py` | 277 | Base runtime builder + FrameworkAdapter + ActivityContext binding |
| `workflow_interceptor.py` | 396 | Workflow events |
| `activity_interceptor.py` | 616 | Activity events |
| `activities.py` | 283 | Governance activity |
| `request_signing.py` | 98 | AIP DID + Ed25519 signed-request construction |
| `multi_agent.py` | 139 | Multi-agent handoff + session-context propagation |
| `tracing.py` | 60 | `@traced` (wraps `openbox_core` `governed()`) + `create_span` |
| **Total** | **~3,800** | **Core SDK** (hook instrumentation lives in `openbox_core`) |

---

## Testing Status

Hook instrumentation internals (HTTP/DB/file/function payload shape, body capture, per-library coverage) are covered by the base SDK's own conformance suite. This package's tests focus on Temporal lifecycle governance and its parity with the base SDK.

### Test Files

| Test File | Coverage |
|-----------|----------|
| `test_activities.py` | Governance event activity submission |
| `test_activity_interceptor.py` | Activity-level governance, redaction, approval polling |
| `test_approval_action_precedence.py` | Verdict/approval precedence ordering |
| `test_base_sdk_signing_parity.py` | Signed-request parity with `openbox_core.identity` |
| `test_config.py` | SDK initialization, API key validation, URL security |
| `test_core_conformance_suite.py` | Base-SDK conformance for the Temporal adapter |
| `test_multi_agent.py` | Handoff events + session-context propagation |
| `test_no_import_machinery_in_governed_open_paths.py` | No import machinery re-entering governed `open()` paths |
| `test_plugin.py` | Plugin construction and interceptor wiring |
| `test_plugin_integration.py` | Plugin end-to-end integration |
| `test_public_api_compatibility.py` | Public API surface stability |
| `test_request_signing.py` | AIP DID + Ed25519 request signing |
| `test_signing_config.py` | Signing configuration + signer loading |
| `test_temporal_hook_parity.py` | Temporal hook governance parity with the base SDK |
| `test_types.py` | Type definitions and verdict conversions |
| `test_worker.py` | Worker factory and setup flow |
| `test_workflow_interceptor.py` | Workflow lifecycle event capture, signal verdicts |
| `test_workflow_sandbox_import_safety.py` | Workflow-sandbox import safety (no forbidden eager imports) |

### Test Coverage Areas

- Type conversions and verdict parsing (v1.0/v1.1 compatibility)
- Temporal hook governance parity with the base SDK (`test_temporal_hook_parity.py`, `test_core_conformance_suite.py`)
- Guardrails input/output redaction (dataclass and dict)
- Configuration validation and API key format checks
- HITL approval polling with expiration handling
- Error policies (fail_open vs fail_closed)
- AIP DID + Ed25519 request signing and byte-for-byte parity with the base SDK
- Signal-verdict bridging and completed-hook stop/halt propagation
- Workflow-sandbox import safety and Temporal determinism compliance

---

## Common Pitfalls

### 1. Module-Level Imports in Workflow Code
**Problem:** Importing `httpx`, `datetime`, or `logging` at module level triggers Temporal sandbox violations
**Solution:** Lazy imports in functions, or import only in activity context

### 2. Forgetting to Add send_governance_event Activity
**Problem:** Workflow interceptor calls activity that doesn't exist
**Solution:** Use `create_openbox_worker()` which adds it automatically, or manually add to activities list

### 3. Hook Governance / Body Capture Not Working
**Problem:** HTTP/DB/file operations aren't governed or bodies aren't captured
**Solution:** Governance instrumentation is installed automatically by `create_openbox_worker()` / `OpenBoxPlugin` (they build an `openbox_core` runtime and call `install_instrumentation()`). The hook payload shape, body-capture rules, and enforcement are owned by `openbox_core` — configure those via the base SDK, not here.

### 4. Stale Verdicts After Workflow Restart
**Problem:** BLOCK verdict from previous run affects new run
**Solution:** Signal verdicts are run-scoped in `TemporalGovernanceState`; a verdict with a mismatched `run_id` is ignored and cleared automatically.

### 5. Approval Never Expires
**Problem:** `approval_expiration_time` not checked or parsed incorrectly
**Solution:** SDK parses ISO 8601 timestamps and compares against UTC time

### 6. Hook Governance Not Blocking Operations
**Problem:** An HTTP request proceeds even though a hook returned BLOCK
**Solution:** Ensure the worker/plugin was created via `create_openbox_worker()` / `OpenBoxPlugin` so the base runtime is installed. Enforcement (blocking/halting) is applied by `openbox_core` and mapped to Temporal effects by `TemporalFrameworkAdapter`; there is no Temporal-local hook config to set.

---

**Document Version:** 1.4
**Last Updated:** 2026-07-03
