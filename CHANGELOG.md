# Changelog

All notable changes to OpenBox SDK for Temporal Workflows.

## [Unreleased] - openbox-core base SDK migration branch

### Changed

- **Depends on `openbox-sdk-python>=0.1.0`** (import package `openbox_core`) — the shared
  base SDK owning contracts, the always-strict gate, identity/signing, the evaluate
  client, context runtime, Core `SpanData` wire serialization, generic instrumentation,
  and the conformance kit.
- `request_signing` now shims over `openbox_core.identity`. Signed request bytes are
  UNCHANGED — gated byte-for-byte by a golden fixture generated from the pre-migration
  signer (`tests/test_base_sdk_signing_parity.py`).
- `Verdict`, `GuardrailsCheckResult`, and `WorkflowEventType` re-export the shared
  contracts; `GovernanceVerdictResponse` subclasses the shared `EvaluationResult`
  (same public surface, plus `fallback_used`/`diagnostics`/`raw`/`approval_expiration_time`).
- Workflow lifecycle events route through shared `EventEnvelope` factories
  (sandbox-safe pure contracts; wire payloads unchanged).
- Activity execution binds the shared `ActivityContext` with a guaranteed
  try/finally reset — governance context can no longer leak when an activity raises.

### Behavior changes (regression-gated)

- **Approval decision precedence:** approval poll responses parse via the shared
  `ApprovalResult`; `action` now outranks `verdict` when both are present
  (previously verdict-first), and a response with NEITHER field stays PENDING
  (previously implicit ALLOW). Pinned by `tests/test_approval_action_precedence.py`.

### Added

- `openbox.core_adapter` — `TemporalFrameworkAdapter` (maps base-SDK verdicts to
  native `ApplicationError` types and the HITL retry loop), `create_core_runtime`
  (opt-in core runtime at worker scope), shared context-store helpers.
- Gates: workflow-sandbox import-safety, base-SDK conformance suite driven by the
  Temporal adapter, public-API compatibility suite.

### Migration status

- Legacy in-repo hook instrumentation remains the DEFAULT. The per-operation flip to
  core instrumentation (HTTP first, then DB/file/function) is gated on hook-payload
  parity; Redis/Mongo/psycopg2-direct stay on legacy hooks until the base SDK scopes
  them. Duplicated modules are removed only after parity.

## [1.1.2] - 2026-04-22

### Security

- **API key no longer flows through workflow history.** `send_governance_event` refactored into a `GovernanceActivities` class that holds credentials on `self`; activity inputs carry only `payload`, `timeout`, and `on_api_error`. Before this change, anyone with Describe permissions on the namespace could read the API key from the recorded activity input.

### Added

- W3C trace propagation through Temporal headers, on by default via `enable_trace_propagation=True` on `OpenBoxPlugin` and `create_openbox_worker()`. Uses `temporalio.contrib.opentelemetry.TracingInterceptor` so spans started by the caller stitch to workflow/activity spans on the worker side.
- `ApplicationError` type constants in `errors.py` (`GOVERNANCE_HALT_ERROR_TYPE`, `GOVERNANCE_BLOCK_ERROR_TYPE`, `GOVERNANCE_API_ERROR_TYPE`, `GOVERNANCE_STOP_ERROR_TYPE`) — single source of truth for governance error routing.
- `openbox.activities.GovernanceActivities` class + `build_governance_activities()` factory.

### Fixed

- **Workflow exception shadowing** — `WorkflowFailed` event send is now wrapped in `try/except`, so a `GovernanceHaltError` from `fail_closed + API down` no longer replaces the real workflow error.
- **String-matching on exception types** — `workflow_interceptor` now inspects `ApplicationError.type` via the exception chain instead of `"GovernanceHalt" in str(e)`. Eliminates false positives when a user workflow happens to emit an error message containing a governance keyword.
- **Race creating governance HTTP client** — `hook_governance._get_sync_client` / `_get_async_client` now use double-checked locking. Previously two concurrent activities could each create a client, with the losing instance getting garbage-collected while its connection pool leaked.
- **Replayer plugin coverage** — `test_plugin_integration.py` now passes `plugins=[plugin]` to `Replayer`, so replay tests validate interceptor determinism, not just user workflow code.
- Version-pin mismatch — comments in `openbox/__init__.py` said `temporalio >= 1.24.0` but pyproject/README pin `1.23.0`. Corrected to `1.23.0`.

### Changed

- `plugin.py` / `worker.py` now use `logging.getLogger(__name__)` for initialization status messages instead of `print()`.

## [1.1.1] - 2026-04-07

### Added

- **OpenBoxPlugin** — drop-in `SimplePlugin` integration for Temporal Workers. Single-line setup: `plugins=[OpenBoxPlugin(openbox_url=..., openbox_api_key=...)]`. Auto-registers interceptors, OTel instrumentation, sandbox passthrough, and `send_governance_event` activity
- Plugin integration guide for Temporal AI Partner Ecosystem (`docs/temporal-plugin-integration-guide.md`)
- HTTP body truncation tests (`tests/test_http_body_truncation.py`)
- Plugin unit tests (`tests/test_plugin.py`) and integration/replay tests (`tests/test_plugin_integration.py`)

### Fixed

- **HTTP body truncation** — enforce `max_body_size` (default 64KB) on request/response bodies in governance spans
- **File I/O spans** — remove raw file content from governance payloads; only `bytes_read`/`bytes_written` metadata sent
- **error_type sanitization** — prevent serialized error objects from being sent as `error.cause.error_type` string in WorkflowFailed payloads
- Remove useless f-strings, redundant `(ImportError, Exception)` clauses, merge nested if statements
- Prefix unused `span` param in urllib hook

### Changed

- `temporalio>=1.23.0` (from 1.8.0) for SimplePlugin support
- `GovernanceConfig.max_body_size` default changed from `None` (unlimited) to `65536` (64KB)
- `\w` regex shorthand in API key pattern

### Refactored

- Reduce cognitive complexity across 7 modules: `activity_interceptor.py` (126→split), `workflow_interceptor.py` (40→split), `activities.py` (20→split), `db_governance_hooks.py` (34→split), `otel_setup.py` (51→split), `tracing.py` (85→split), `verdict_handler.py` (16→15)
- Extract shared helpers: `_run_governed_query_sync/async`, `_build_error_dict`, `_extract_dbapi_context`, `_instrument_sqlalchemy`

### Dependencies

- Bump Pygments 2.19.2 → 2.20.0 (ReDoS fix, CVSS 1.9)

## [1.1.0] - 2026-03-09

### Added

- **Hook-level governance** — real-time, per-operation governance evaluation during activity execution
  - Every HTTP request, database query, file operation, and traced function call is evaluated at `started` (before, can block) and `completed` (after, informational) stages
  - Same `POST /api/v1/governance/evaluate` endpoint with new `hook_trigger` field in payload
  - Automatically enabled when using `create_openbox_worker()`
- **Database query governance** — per-query started/completed evaluations for psycopg2, pymysql, mysql-connector, asyncpg, pymongo, redis, sqlalchemy
- **File I/O governance** — per-operation evaluations for open, read, write, readline, readlines, writelines, close (opt-in via `instrument_file_io=True`)
- **`@traced` decorator** (`openbox.tracing`) — function-level governance with OTel spans; zero overhead when governance not configured
- **`GovernanceBlockedError`** — new exception type for hook-level blocking with verdict, reason, and resource identifier
- **Abort propagation** — once one hook blocks, all subsequent hooks for the same activity short-circuit immediately
- **HALT workflow termination** from hook-level governance via `client.terminate()`
- **REQUIRE_APPROVAL** from hook-level governance enters the same HITL approval polling flow as activity-level approvals
- **`duration_ns`** computed for all hook span types (HTTP, file, function — DB already had it)

### Changed

- **`hook_trigger` simplified to boolean** — was a dict with type/stage/data, now just `true`. All data moved to span root fields
- **Span data consolidation** — all type-specific fields at span root (`hook_type`, `http_method`, `db_system`, `file_path`, `function`, etc.)
- **`attributes` is OTel-original only** — no custom `openbox.*`, `http.request.*`, `db.result.*` fields injected
- Hook governance payloads send only the current span per evaluation (not accumulated history)
- Event-level payloads (ActivityStarted/Completed, Workflow events) no longer include spans
- Simplified `WorkflowSpanProcessor` — removed span buffering, governed span tracking, body data merging; `on_end()` now only forwards to fallback exporters

### Fixed

- HALT verdict from hooks now correctly terminates the workflow (previously only stopped the activity like BLOCK)
- REQUIRE_APPROVAL from hooks now enters the approval polling flow (previously raised unhandled error)
- Stale buffer/verdict from previous workflow run no longer carries over when workflow_id is reused
- Subsequent hooks no longer fire after the first hook blocks an activity

## [1.0.21] - 2026-03-04

### Added

- Human-in-the-loop approval with expiration handling
- Approval polling via `POST /api/v1/governance/approval`
- Guardrails: input/output validation and redaction
- `GovernanceVerdictResponse.from_dict()` with guardrails_result parsing
- Output redaction for activity results
- `_deep_update_dataclass()` for in-place dataclass field updates from redacted dicts

### Fixed

- Temporal Payload objects no longer slip through as non-serializable in governance payloads
- Stale buffer detection via run_id comparison

## [1.0.0] - 2026-02-15

### Added

- Initial release
- 6 event types: WorkflowStarted, WorkflowCompleted, WorkflowFailed, SignalReceived, ActivityStarted, ActivityCompleted
- 5-tier verdict system: ALLOW, CONSTRAIN, REQUIRE_APPROVAL, BLOCK, HALT
- HTTP instrumentation via OpenTelemetry (httpx, requests, urllib3, urllib)
- Database instrumentation (psycopg2, pymysql, asyncpg, pymongo, redis, sqlalchemy)
- File I/O instrumentation (opt-in)
- Zero-code setup via `create_openbox_worker()` factory
- Workflow and activity interceptors for governance
- Span buffering and activity context tracking
- `fail_open` / `fail_closed` error policies
- v1.0 backward compatibility for legacy verdict strings
