# Changelog

All notable changes to OpenBox SDK for Temporal Workflows.

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
