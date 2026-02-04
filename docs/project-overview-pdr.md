# Project Overview & Product Development Requirements (PDR)

**Project Name:** OpenBox SDK for Temporal Workflows
**Version:** 1.0.0
**Status:** Alpha
**Last Updated:** 2026-02-04
**Total LOC:** 3,583 (across 10 Python files)

---

## Executive Summary

OpenBox SDK for Temporal Workflows is a Python SDK that provides **workflow-boundary governance and observability** for Temporal-based applications. It captures workflow/activity lifecycle events, HTTP telemetry (request/response bodies and headers), and sends them to OpenBox Core for policy evaluation. The SDK enables graduated governance responses (ALLOW, CONSTRAIN, REQUIRE_APPROVAL, BLOCK, HALT) and guardrails for input/output validation and redaction.

### Core Value Proposition

- **Zero-code instrumentation** via `create_openbox_worker()` factory function
- **Comprehensive telemetry capture**: workflow events, activity I/O, HTTP calls, database queries, file operations
- **Graduated governance verdicts**: 5-tier response (ALLOW → CONSTRAIN → REQUIRE_APPROVAL → BLOCK → HALT)
- **Human-in-the-loop (HITL) support**: Pause workflows for human approval with expiration handling
- **Guardrails system**: Validate and redact sensitive data before/after execution
- **Fail-safe policies**: Configurable fail-open/fail-closed behavior when governance API is unreachable

---

## Product Development Requirements

### 1. Functional Requirements

#### FR-1: Workflow Lifecycle Event Capture
- **Requirement**: Capture all workflow lifecycle events for governance evaluation
- **Events**:
  - `WorkflowStarted`: Workflow begins execution (optional, configurable)
  - `WorkflowCompleted`: Workflow succeeds with output
  - `WorkflowFailed`: Workflow fails with error details
  - `SignalReceived`: Workflow receives a signal with arguments
- **Implementation**: `GovernanceInterceptor` in `workflow_interceptor.py`
- **Acceptance Criteria**:
  - Events sent via activity for workflow determinism
  - Timestamps in RFC3339 format (UTC)
  - Error details include nested cause chain

#### FR-2: Activity Lifecycle Event Capture
- **Requirement**: Capture activity execution with input/output data and HTTP spans
- **Events**:
  - `ActivityStarted`: Activity begins with input arguments (optional, configurable)
  - `ActivityCompleted`: Activity ends with output, status, duration, and captured spans
- **Implementation**: `ActivityGovernanceInterceptor` in `activity_interceptor.py`
- **Acceptance Criteria**:
  - Captures positional and keyword arguments
  - Serializes dataclass arguments to JSON
  - Includes all child HTTP/database/file spans
  - Events sent directly via HTTP (activities are non-deterministic)

#### FR-3: HTTP Telemetry Capture
- **Requirement**: Capture request/response bodies and headers for HTTP calls made within activities
- **Supported Libraries**:
  - `httpx` (sync + async) - full body capture via patching
  - `requests` - full body capture via hooks
  - `urllib3` - full body capture via hooks
  - `urllib` - request body only
- **Implementation**: `otel_setup.py` with instrumentation hooks and `WorkflowSpanProcessor` buffering
- **Acceptance Criteria**:
  - Only text content types captured (JSON, XML, text/*)
  - Bodies stored separately from OTel spans (privacy)
  - Ignored URLs configurable (e.g., OpenBox Core API)

#### FR-4: Database Query Capture
- **Requirement**: Capture database queries as spans for policy evaluation
- **Supported Databases**:
  - PostgreSQL (`psycopg2`, `asyncpg`)
  - MySQL (`mysql-connector-python`, `pymysql`)
  - MongoDB (`pymongo`)
  - Redis (`redis`)
  - SQLAlchemy ORM
- **Implementation**: `setup_database_instrumentation()` in `otel_setup.py`
- **Acceptance Criteria**:
  - Query statement captured in `db.statement` attribute
  - Database system identified in `db.system` attribute
  - Connection details (host, port, database name) included
  - Enabled by default, configurable via `instrument_databases` flag

#### FR-5: File I/O Capture (Optional)
- **Requirement**: Capture file operations as spans
- **Operations**: `open()`, `read()`, `write()`, `readline()`, `readlines()`, `writelines()`
- **Implementation**: Monkey-patches `builtins.open` in `otel_setup.py`
- **Acceptance Criteria**:
  - File path, mode, operation type, and bytes read/written captured
  - System paths skipped (`/dev/`, `/proc/`, `/sys/`, `__pycache__`)
  - Disabled by default (noisy), opt-in via `instrument_file_io=True`

#### FR-6: Function Tracing Decorator
- **Requirement**: Allow developers to trace custom functions as spans
- **Usage**: `@traced` decorator in `tracing.py`
- **Acceptance Criteria**:
  - Supports sync and async functions
  - Captures function arguments and return values
  - Configurable argument length truncation (default: 2000 chars)
  - Exception details captured on error

#### FR-7: Governance Verdict Handling
- **Requirement**: Process governance verdicts and enforce actions
- **5-Tier Verdict System**:
  1. `ALLOW` - Continue execution normally
  2. `CONSTRAIN` - Log constraints, continue (future sandbox enforcement)
  3. `REQUIRE_APPROVAL` - Pause and poll for human approval
  4. `BLOCK` - Raise non-retryable error, stop activity
  5. `HALT` - Raise non-retryable error, terminate workflow
- **Implementation**: `Verdict` enum in `types.py`, enforced in interceptors
- **Acceptance Criteria**:
  - Verdicts have priority ordering (HALT > BLOCK > REQUIRE_APPROVAL > CONSTRAIN > ALLOW)
  - `BLOCK`/`HALT` raise `ApplicationError` with `non_retryable=True`
  - Backward compatible with v1.0 action strings (`continue`, `stop`, `require-approval`)

#### FR-8: Guardrails System
- **Requirement**: Validate and redact sensitive data in activity input/output
- **Features**:
  - **Pre-execution**: Redact activity input before it executes
  - **Post-execution**: Redact activity output before it returns
  - **Validation**: Block execution if `validation_passed=false`
- **Implementation**: `GuardrailsCheckResult` in `types.py`, applied in `activity_interceptor.py`
- **Acceptance Criteria**:
  - Supports dataclass and dict redaction
  - Deep recursive field updates for nested dataclasses
  - Validation failures raise `ApplicationError` with type `GuardrailsValidationFailed`
  - Reasons array provides detailed failure information

#### FR-9: Human-in-the-Loop (HITL) Approval
- **Requirement**: Pause workflow execution for human approval on `REQUIRE_APPROVAL` verdict
- **Behavior**:
  - Activity raises retryable `ApplicationError` with type `ApprovalPending`
  - On retry, polls OpenBox Core for approval status via `/api/v1/governance/approval` endpoint
  - If approved (`verdict=allow`), clears pending flag and proceeds
  - If rejected (`verdict=block/halt`), raises non-retryable error
  - If expired, terminates workflow immediately
- **Implementation**: `_poll_approval_status()` in `activity_interceptor.py`
- **Acceptance Criteria**:
  - Pending approval stored in `WorkflowSpanBuffer.pending_approval`
  - Approval expiration time checked against current UTC time
  - Expired approvals terminate with `ApprovalExpired` error type
  - Configurable via `hitl_enabled` flag (default: True)

#### FR-10: Worker Factory Function
- **Requirement**: Provide simple factory function for zero-code setup
- **Function**: `create_openbox_worker()` in `worker.py`
- **Acceptance Criteria**:
  - Validates API key format and connectivity
  - Sets up span processor and OTel instrumentation
  - Creates workflow + activity interceptors
  - Returns fully configured `Worker` instance
  - Accepts all standard Temporal Worker parameters

---

### 2. Non-Functional Requirements

#### NFR-1: Performance
- **Requirement**: Minimal overhead on workflow execution
- **Targets**:
  - HTTP body capture: <10ms per request
  - Span buffering: O(1) registration, O(n) retrieval
  - Event serialization: <5ms for typical payloads
- **Constraints**:
  - Bodies stored separately from OTel spans to avoid exporter overhead
  - Ignored URLs skip instrumentation entirely

#### NFR-2: Reliability
- **Requirement**: Fail-safe governance with configurable error handling
- **Policies**:
  - `fail_open` (default): Allow workflow to continue if governance API is unreachable
  - `fail_closed`: Halt workflow if governance API is unreachable
- **Implementation**: `on_api_error` in `GovernanceConfig`
- **Acceptance Criteria**:
  - Network errors logged but don't crash worker
  - Governance API failures respect policy setting
  - Timeout configurable (default: 30s)

#### NFR-3: Security
- **Requirement**: Protect sensitive data from exposure
- **Measures**:
  - Bodies stored in span processor buffer, not exported to external tracing systems
  - API key validated on initialization
  - Binary content types never captured
  - Ignored URL prefixes prevent governance event loops
- **API Key Format**: `obx_live_*` or `obx_test_*`

#### NFR-4: Compatibility
- **Requirement**: Support Python 3.9+ and Temporal SDK 1.8+
- **Dependencies**:
  - Core: `temporalio>=1.8.0`, `httpx>=0.28.0`
  - OTel: `opentelemetry-api>=1.38.0`, `opentelemetry-sdk>=1.38.0`
  - Instrumentation: `opentelemetry-instrumentation-*>=0.59b0`
- **Python Versions**: 3.9, 3.10, 3.11, 3.12

#### NFR-5: Temporal Sandbox Compliance
- **Requirement**: Avoid Temporal determinism violations
- **Constraints**:
  - Workflow interceptor sends events via activity (no direct HTTP)
  - Activity interceptor sends events directly (non-deterministic allowed)
  - No module-level `datetime`/`time` imports in workflow code
  - No module-level `logging` imports in workflow code (uses `linecache` → `os.stat`)
  - Lazy imports for `httpx`, `datetime`, `logging` in activities
- **Implementation**: Strict separation of workflow vs activity code paths

#### NFR-6: Observability
- **Requirement**: Provide debugging and monitoring capabilities
- **Features**:
  - Structured logging with `activity.logger` in activities
  - Event payloads include timestamps, workflow IDs, activity IDs
  - Span data includes trace IDs for correlation
  - Error details include full exception chain

---

### 3. Technical Constraints

#### TC-1: Temporal Determinism
- **Constraint**: Workflow code must be deterministic (no HTTP, no datetime, no os.stat)
- **Impact**: Workflow interceptor uses activity for HTTP calls
- **Mitigation**: `send_governance_event` activity handles all workflow-level HTTP

#### TC-2: OpenTelemetry Sandbox Restrictions
- **Constraint**: OTel uses `importlib_metadata` which uses `os.stat` (sandbox violation)
- **Impact**: Cannot import OTel at module level in workflow files
- **Mitigation**: Lazy imports in activities, not re-exported from `openbox/__init__.py`

#### TC-3: HTTP Body Capture Limitations
- **Constraint**: Some HTTP clients consume streams (cannot re-read)
- **Impact**: `httpx` response hooks may not capture body if stream not read
- **Mitigation**: Patch `Client.send` to capture bodies at send time

#### TC-4: Verdict Staleness
- **Constraint**: Workflow run IDs change on continue-as-new
- **Impact**: Verdicts from previous run should not affect new run
- **Mitigation**: Store `run_id` with verdict, clear stale verdicts on mismatch

---

### 4. Integration Points

#### INT-1: OpenBox Core API
- **Base URL**: Configurable (e.g., `http://localhost:8086`)
- **Endpoints**:
  - `POST /api/v1/governance/evaluate` - Evaluate governance event, return verdict
  - `POST /api/v1/governance/approval` - Poll approval status for HITL
  - `GET /api/v1/auth/validate` - Validate API key on initialization
- **Authentication**: Bearer token (`Authorization: Bearer {api_key}`)
- **Response Format**: JSON with `verdict`, `reason`, `policy_id`, `risk_score`, `guardrails_result`, `approval_expiration_time`

#### INT-2: Temporal Server
- **Connection**: Via `temporalio.client.Client`
- **Version**: Temporal SDK 1.8+
- **Interceptor Chain**: `GovernanceInterceptor` → `ActivityGovernanceInterceptor` → user interceptors

#### INT-3: OpenTelemetry
- **TracerProvider**: SDK creates and registers `WorkflowSpanProcessor`
- **Instrumentors**: HTTP, database, file I/O (via OTel community packages)
- **Span Export**: Optional fallback processor for external tracing systems

---

### 5. Success Metrics

#### SM-1: Adoption Metrics
- **Zero-code setup**: >80% of users use `create_openbox_worker()` factory
- **Instrumentation coverage**: All 6 event types captured
- **HITL adoption**: >50% of users enable approval polling

#### SM-2: Performance Metrics
- **Latency**: <50ms p95 for governance API calls
- **Overhead**: <5% CPU increase with full instrumentation
- **Memory**: <100MB additional memory per worker

#### SM-3: Reliability Metrics
- **Uptime**: 99.9% governance API availability
- **Error rate**: <0.1% governance evaluation failures
- **Verdict enforcement**: 100% compliance (no HALT verdicts bypassed)

---

### 6. Future Enhancements

#### FE-1: Constraint Enforcement
- **Description**: Enforce `CONSTRAIN` verdict by modifying activity behavior (e.g., rate limiting, sandboxing)
- **Priority**: Medium
- **Timeline**: Q3 2026

#### FE-2: Approval UI
- **Description**: Embedded approval UI in OpenBox Dashboard
- **Priority**: High
- **Timeline**: Q2 2026

#### FE-3: Policy DSL
- **Description**: Allow users to define governance policies in SDK (not just server-side)
- **Priority**: Low
- **Timeline**: Q4 2026

#### FE-4: Replay Protection
- **Description**: Detect and handle Temporal workflow replay scenarios
- **Priority**: Medium
- **Timeline**: Q3 2026

---

### 7. Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Temporal sandbox violations | High | Medium | Strict lazy imports, automated testing |
| OTel instrumentation conflicts | Medium | Low | Version pinning, compatibility matrix |
| Governance API downtime | High | Medium | Fail-open default, configurable timeout |
| Body capture breaking HTTP clients | Medium | Low | Defensive programming, skip binary content |
| Verdict staleness across runs | Medium | Medium | Store run_id with verdict, clear on mismatch |

---

### 8. Dependencies

#### Core Dependencies
- `temporalio>=1.8.0,<2` - Temporal Python SDK
- `httpx>=0.28.0,<1` - HTTP client with async support
- `opentelemetry-api>=1.38.0` - OTel API for tracing
- `opentelemetry-sdk>=1.38.0` - OTel SDK for span processing

#### HTTP Instrumentation
- `opentelemetry-instrumentation-httpx>=0.59b0`
- `opentelemetry-instrumentation-requests>=0.59b0`
- `opentelemetry-instrumentation-urllib3>=0.59b0`
- `opentelemetry-instrumentation-urllib>=0.59b0`

#### Database Drivers & Instrumentation
- `psycopg2-binary>=2.9.10` + `opentelemetry-instrumentation-psycopg2>=0.59b0`
- `asyncpg>=0.29.0` + `opentelemetry-instrumentation-asyncpg>=0.59b0`
- `mysql-connector-python>=8.0.0` + `opentelemetry-instrumentation-mysql>=0.59b0`
- `pymysql>=1.0.0` + `opentelemetry-instrumentation-pymysql>=0.59b0`
- `pymongo>=4.0.0` + `opentelemetry-instrumentation-pymongo>=0.59b0`
- `redis>=5.0.0` + `opentelemetry-instrumentation-redis>=0.59b0`
- `sqlalchemy>=2.0.0` + `opentelemetry-instrumentation-sqlalchemy>=0.59b0`

---

### 9. Glossary

- **Verdict**: Governance decision returned by OpenBox Core (ALLOW, CONSTRAIN, REQUIRE_APPROVAL, BLOCK, HALT)
- **Guardrails**: Input/output validation and redaction system
- **HITL**: Human-in-the-loop approval system for sensitive operations
- **Span**: OpenTelemetry distributed trace unit representing an operation
- **Determinism**: Temporal requirement that workflow code produces same result on replay
- **Fail-open**: Continue execution when governance API is unavailable
- **Fail-closed**: Stop execution when governance API is unavailable

---

**Document Version**: 1.0
**Prepared By**: OpenBox Documentation System
**Date**: 2026-02-04
**Test Status**: 10 test files implemented (see `./tests/` directory)
