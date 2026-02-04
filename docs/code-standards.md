# Code Standards & Best Practices

**Last Updated:** 2026-02-04

---

## Overview

This document defines coding standards, architectural patterns, and best practices for the OpenBox SDK for Temporal Workflows. All contributors must adhere to these standards to maintain code quality and consistency.

---

## Project Architecture

### Layer Separation

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Code                               │
│  (Workflows, Activities, Worker Setup)                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Public API Layer                             │
│  - create_openbox_worker()                                      │
│  - initialize()                                                 │
│  - GovernanceConfig, Verdict, WorkflowEventType                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Interceptor Layer                             │
│  - GovernanceInterceptor (workflow-safe)                        │
│  - ActivityGovernanceInterceptor (activity-only)                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Instrumentation Layer                           │
│  - WorkflowSpanProcessor (OTel span buffering)                  │
│  - otel_setup (HTTP/DB/File instrumentation)                    │
│  - tracing (@traced decorator)                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Integration Layer                             │
│  - OpenBox Core API (REST)                                      │
│  - Temporal Server (gRPC)                                       │
│  - OpenTelemetry (Tracing)                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Temporal-Specific Standards

### Rule 1: Workflow Determinism

**CRITICAL:** Workflow code MUST be deterministic.

#### Prohibited in Workflow Context
- ❌ Direct HTTP calls (`httpx`, `requests`, `urllib`)
- ❌ Datetime operations (`datetime.now()`, `time.time()`)
- ❌ Random number generation (`random.random()`)
- ❌ File I/O (`open()`, `os.path.exists()`)
- ❌ Module-level imports of non-deterministic libraries
- ❌ `os.stat`, `os.path.exists`, `os.listdir`
- ❌ Global state mutations

#### Allowed in Workflow Context
- ✅ `workflow.execute_activity()` - delegate to activities
- ✅ `workflow.sleep()` - deterministic delay
- ✅ `workflow.patched()` - version gates
- ✅ Pure functions with deterministic logic
- ✅ Temporal SDK utilities (`workflow.info()`, `workflow.uuid4()`)

#### Example: Workflow Interceptor

```python
# ✅ CORRECT - Sends events via activity
async def execute_workflow(self, input: ExecuteWorkflowInput) -> Any:
    info = workflow.info()

    # Send event via activity (deterministic)
    await workflow.execute_activity(
        "send_governance_event",
        args=[{"api_url": api_url, "payload": {...}}],
        start_to_close_timeout=timedelta(seconds=30),
    )

    result = await super().execute_workflow(input)
    return result
```

```python
# ❌ WRONG - Direct HTTP call
async def execute_workflow(self, input: ExecuteWorkflowInput) -> Any:
    import httpx  # ❌ Non-deterministic import

    # ❌ Direct HTTP call in workflow
    async with httpx.AsyncClient() as client:
        await client.post(api_url, json={...})

    result = await super().execute_workflow(input)
    return result
```

---

### Rule 2: Temporal Sandbox Restrictions

**CRITICAL:** Avoid `os.stat` and related syscalls in workflow imports.

#### Problem Libraries
- `httpx` - Uses `anyio` → `os.stat`
- `datetime` (module-level) - Uses `_strptime` → `os.stat`
- `logging` (module-level) - Uses `linecache` → `os.stat`
- `opentelemetry` - Uses `importlib_metadata` → `os.stat`

#### Solution: Lazy Imports

```python
# ✅ CORRECT - Lazy import in activity function
@activity.defn(name="send_governance_event")
async def send_governance_event(input: Dict[str, Any]):
    import httpx  # ✅ Loaded only when activity executes
    from datetime import datetime, timezone  # ✅ Lazy import

    async with httpx.AsyncClient() as client:
        response = await client.post(...)
    return response.json()
```

```python
# ❌ WRONG - Module-level import
import httpx  # ❌ Loaded when module imports, triggers sandbox

@activity.defn(name="send_governance_event")
async def send_governance_event(input: Dict[str, Any]):
    async with httpx.AsyncClient() as client:
        response = await client.post(...)
    return response.json()
```

#### Lazy Logger Pattern

```python
# ✅ CORRECT - Lazy logger to avoid sandbox
def _get_logger():
    """Lazy logger to avoid sandbox restrictions."""
    import logging
    return logging.getLogger(__name__)

# Usage
_get_logger().info("Message")
```

```python
# ❌ WRONG - Module-level logger
import logging  # ❌ Uses linecache → os.stat
logger = logging.getLogger(__name__)

logger.info("Message")  # Triggers sandbox violation
```

---

### Rule 3: Workflow-Safe Public API

**CRITICAL:** `openbox/__init__.py` must only export workflow-safe modules.

#### Workflow-Safe Modules
- ✅ `types.py` - Pure Python dataclasses and enums
- ✅ `config.py` - Lazy imports for urllib/logging
- ✅ `workflow_interceptor.py` - No direct HTTP
- ✅ `span_processor.py` - No external dependencies

#### Activity-Only Modules (NOT re-exported)
- ❌ `activities.py` - Uses `httpx`
- ❌ `activity_interceptor.py` - Uses `opentelemetry`
- ❌ `otel_setup.py` - Uses `opentelemetry`
- ❌ `tracing.py` - Uses `opentelemetry`

#### Public API Pattern

```python
# openbox/__init__.py

# ✅ Safe to export (workflow-safe)
from .types import Verdict, WorkflowEventType
from .workflow_interceptor import GovernanceInterceptor

# ❌ NOT exported (uses OTel/httpx)
# from .activity_interceptor import ActivityGovernanceInterceptor
# from .activities import send_governance_event

# Users must import directly:
# from openbox.activity_interceptor import ActivityGovernanceInterceptor
```

---

## Python Standards

### Code Style

**Formatter:** Black 23.7+ (current: 23.7.0)
**Import Sorter:** isort 5.12+ (black profile, current: 5.12.0)
**Type Checker:** mypy 1.16+ (Python 3.9 target, current: 1.16.0)

#### Style Rules
- Line length: 100 characters (Black default: 88, relaxed to 100)
- Indentation: 4 spaces
- Quotes: Double quotes (Black default)
- Trailing commas: Yes (Black enforced)

#### Example

```python
from typing import Dict, List, Optional

def process_data(
    input_data: List[Dict[str, Any]],
    config: Optional[GovernanceConfig] = None,
) -> Dict[str, Any]:
    """
    Process input data with governance config.

    Args:
        input_data: List of data dictionaries to process
        config: Optional governance configuration

    Returns:
        Processed data dictionary
    """
    results = []
    for item in input_data:
        processed = transform(item)
        results.append(processed)

    return {"results": results, "count": len(results)}
```

---

### Type Annotations

**Standard:** PEP 484 type hints required for all public APIs

#### Required Annotations
- ✅ Function parameters
- ✅ Function return types
- ✅ Class attributes (dataclasses)
- ✅ Module-level constants

#### Optional Annotations
- Local variables (use if improves clarity)
- Private methods (recommended but not required)

#### Example

```python
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field

@dataclass
class GovernanceConfig:
    """Configuration for governance interceptors."""

    skip_workflow_types: Set[str] = field(default_factory=set)
    on_api_error: str = "fail_open"
    api_timeout: float = 30.0

def create_openbox_worker(
    client: Client,
    task_queue: str,
    *,
    workflows: Sequence[Type] = (),
    activities: Sequence[Callable] = (),
    openbox_url: Optional[str] = None,
    openbox_api_key: Optional[str] = None,
) -> Worker:
    """Create a Temporal Worker with OpenBox governance enabled."""
    pass
```

---

### Error Handling

#### Exception Hierarchy

```python
OpenBoxConfigError (base)
├── OpenBoxAuthError (invalid API key)
└── OpenBoxNetworkError (connectivity issues)

ApplicationError (Temporal)
├── GovernanceStop (BLOCK/HALT verdicts)
├── GuardrailsValidationFailed (guardrails failure)
├── ApprovalPending (awaiting approval, retryable)
├── ApprovalExpired (approval expired, non-retryable)
└── ApprovalRejected (approval denied, non-retryable)
```

#### Non-Retryable vs Retryable

```python
# ✅ Non-retryable - Terminates workflow immediately
raise ApplicationError(
    "Governance blocked: Policy violation",
    type="GovernanceStop",
    non_retryable=True,  # ✅ No retry
)

# ✅ Retryable - Temporal will retry with backoff
raise ApplicationError(
    "Awaiting approval",
    type="ApprovalPending",
    non_retryable=False,  # ✅ Retryable
)
```

#### Error Logging

```python
# ✅ CORRECT - Activity logger (safe outside sandbox)
activity.logger.info("Processing activity")
activity.logger.warning(f"API error: {error}")
activity.logger.error("Failed to process", exc_info=True)

# ✅ CORRECT - Lazy logger in non-activity code
def _get_logger():
    import logging
    return logging.getLogger(__name__)

_get_logger().info("Message")

# ❌ WRONG - Module-level logger in workflow code
import logging
logger = logging.getLogger(__name__)  # Triggers sandbox
logger.info("Message")
```

---

## OpenTelemetry Standards

### Span Naming

**Convention:** `{operation}.{resource}` or `{library}.{operation}`

#### Examples
- ✅ `activity.call_llm` - Activity span
- ✅ `POST` - HTTP span (method as name)
- ✅ `SELECT` - Database span (operation as name)
- ✅ `file.open`, `file.read`, `file.write` - File I/O spans
- ✅ `process_data` - Function span (from @traced)

### Span Attributes

**Convention:** Follow OpenTelemetry semantic conventions

#### HTTP Spans
```python
{
    "http.method": "POST",
    "http.url": "https://api.example.com/v1/chat",
    "http.status_code": 200,
    "http.target": "/v1/chat",
    "net.peer.name": "api.example.com",
    "net.peer.port": 443,
}
```

#### Database Spans
```python
{
    "db.system": "postgresql",
    "db.name": "mydb",
    "db.statement": "SELECT * FROM users WHERE id = $1",
    "db.operation": "SELECT",
    "net.peer.name": "localhost",
    "net.peer.port": 5432,
}
```

#### File I/O Spans
```python
{
    "file.path": "/app/data/config.json",
    "file.mode": "r",
    "file.operation": "read",
    "file.bytes": 1024,
}
```

#### Temporal Attributes (Custom)
```python
{
    "temporal.workflow_id": "my-workflow-123",
    "temporal.activity_id": "1",
    "temporal.workflow_type": "MyWorkflow",
    "temporal.task_queue": "my-task-queue",
}
```

---

## API Design Standards

### REST API Conventions

**OpenBox Core API endpoints:**

#### POST /api/v1/governance/evaluate
**Purpose:** Evaluate governance event, return verdict

**Request:**
```json
{
  "source": "workflow-telemetry",
  "event_type": "ActivityCompleted",
  "workflow_id": "my-workflow-123",
  "run_id": "abc-123",
  "activity_id": "1",
  "activity_type": "call_llm",
  "activity_input": [{"prompt": "Hello"}],
  "activity_output": {"response": "Hi there"},
  "spans": [...],
  "timestamp": "2024-01-01T00:00:00.000Z"
}
```

**Response:**
```json
{
  "verdict": "allow",
  "reason": "Request approved",
  "policy_id": "policy-123",
  "risk_score": 0.1,
  "guardrails_result": {
    "input_type": "activity_input",
    "redacted_input": {"prompt": "[REDACTED]"},
    "validation_passed": true,
    "reasons": []
  }
}
```

#### POST /api/v1/governance/approval
**Purpose:** Poll approval status for HITL

**Request:**
```json
{
  "workflow_id": "my-workflow-123",
  "run_id": "abc-123",
  "activity_id": "1"
}
```

**Response:**
```json
{
  "verdict": "allow",
  "reason": "Approved by admin",
  "approval_expiration_time": "2024-01-01T01:00:00.000Z"
}
```

---

## Testing Standards

### Unit Test Structure

```python
import pytest
from openbox.types import Verdict, GovernanceVerdictResponse

class TestVerdict:
    """Test Verdict enum and methods."""

    def test_from_string_v1_1(self):
        """Test parsing v1.1 verdict strings."""
        assert Verdict.from_string("allow") == Verdict.ALLOW
        assert Verdict.from_string("halt") == Verdict.HALT

    def test_from_string_v1_0_compat(self):
        """Test backward compatibility with v1.0 action strings."""
        assert Verdict.from_string("continue") == Verdict.ALLOW
        assert Verdict.from_string("stop") == Verdict.HALT

    def test_priority(self):
        """Test verdict priority ordering."""
        assert Verdict.HALT.priority == 5
        assert Verdict.BLOCK.priority == 4
        assert Verdict.REQUIRE_APPROVAL.priority == 3
        assert Verdict.CONSTRAIN.priority == 2
        assert Verdict.ALLOW.priority == 1

    def test_should_stop(self):
        """Test should_stop() method."""
        assert Verdict.HALT.should_stop() is True
        assert Verdict.BLOCK.should_stop() is True
        assert Verdict.ALLOW.should_stop() is False
```

### Integration Test Structure

```python
import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

@pytest.fixture
async def env():
    """Start Temporal test environment."""
    async with await WorkflowEnvironment.start_time_skipping() as env:
        yield env

@pytest.mark.asyncio
async def test_workflow_with_governance(env):
    """Test workflow with governance interceptor."""
    # Setup
    worker = create_openbox_worker(
        client=env.client,
        task_queue="test-queue",
        workflows=[MyWorkflow],
        activities=[my_activity],
        openbox_url="http://mock-server",
        openbox_api_key="obx_test_key",
    )

    # Execute
    async with worker:
        result = await env.client.execute_workflow(
            MyWorkflow.run,
            id="test-workflow",
            task_queue="test-queue",
        )

    # Assert
    assert result == expected_result
```

---

## Documentation Standards

### Docstring Format

**Standard:** Google-style docstrings

```python
def create_openbox_worker(
    client: Client,
    task_queue: str,
    *,
    openbox_url: Optional[str] = None,
    openbox_api_key: Optional[str] = None,
) -> Worker:
    """
    Create a Temporal Worker with OpenBox governance enabled.

    This function validates the API key, sets up OpenTelemetry instrumentation,
    creates governance interceptors, and returns a fully configured Worker.

    Args:
        client: Temporal client instance
        task_queue: Task queue name for the worker
        openbox_url: OpenBox Core API URL (required for governance)
        openbox_api_key: OpenBox API key (format: obx_live_* or obx_test_*)

    Returns:
        Configured Temporal Worker instance with governance enabled

    Raises:
        OpenBoxAuthError: Invalid API key
        OpenBoxNetworkError: Cannot reach OpenBox Core

    Example:
        >>> worker = create_openbox_worker(
        ...     client=client,
        ...     task_queue="my-queue",
        ...     workflows=[MyWorkflow],
        ...     activities=[my_activity],
        ...     openbox_url="http://localhost:8086",
        ...     openbox_api_key="obx_test_key_1",
        ... )
        >>> await worker.run()
    """
    pass
```

### Inline Comments

**Rules:**
- Use comments to explain **WHY**, not **WHAT**
- Mark critical sections with `# IMPORTANT:` or `# CRITICAL:`
- Mark workarounds with `# NOTE:` or `# WORKAROUND:`
- Mark TODOs with `# TODO(author):` format

```python
# ✅ GOOD - Explains WHY
# Lazy import to avoid Temporal sandbox restrictions.
# httpx uses os.stat internally which triggers sandbox errors.
import httpx

# ✅ GOOD - Marks critical section
# IMPORTANT: Workflow code must be deterministic.
# Do NOT make HTTP calls directly - use activities instead.
await workflow.execute_activity(...)

# ❌ BAD - Explains WHAT (obvious from code)
# Call the API
response = await client.post(url)
```

---

## Security Standards

### API Key Handling

**Format:** `obx_live_*` or `obx_test_*`

```python
# ✅ CORRECT - Validate API key format
API_KEY_PATTERN = re.compile(r"^obx_(live|test)_[a-zA-Z0-9_]+$")

def _validate_api_key_format(api_key: str) -> bool:
    """Validate API key format."""
    return bool(API_KEY_PATTERN.match(api_key))

# ✅ CORRECT - Validate with server
def _validate_api_key_with_server(api_url: str, api_key: str):
    """Validate API key by calling /v1/auth/validate endpoint."""
    response = requests.get(
        f"{api_url}/api/v1/auth/validate",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    if response.status_code != 200:
        raise OpenBoxAuthError("Invalid API key")
```

### Sensitive Data Handling

**Rules:**
- ✅ Store HTTP bodies in span processor buffer, NOT in OTel span attributes
- ✅ Only capture text content types (skip binary)
- ✅ Ignore configured URLs (e.g., OpenBox Core API)
- ✅ Redact sensitive fields via guardrails system
- ❌ Never log API keys or tokens
- ❌ Never export sensitive data to external tracing systems

```python
# ✅ CORRECT - Bodies stored separately
_span_processor.store_body(span_id, request_body=body)

# ❌ WRONG - Bodies in OTel attributes (exported!)
span.set_attribute("http.request.body", body)
```

---

## Performance Standards

### Optimization Rules

1. **Lazy Initialization**: Defer expensive operations until needed
2. **Buffer Spans**: Batch span submission, don't send per-span
3. **Skip Ignored URLs**: Early return to avoid instrumentation overhead
4. **Limit Body Size**: Configurable max body size (default: no limit)
5. **Thread-Safe**: Use locks for shared state

### Example: Skip Ignored URLs

```python
def _should_ignore_url(url: str) -> bool:
    """Check if URL should be ignored (early return)."""
    if not url:
        return False
    for prefix in _ignored_url_prefixes:
        if url.startswith(prefix):
            return True  # ✅ Early return
    return False

def _httpx_request_hook(span, request):
    """Hook called before httpx sends a request."""
    # ✅ Early return for ignored URLs
    if _should_ignore_url(str(request.url)):
        return

    # Expensive body capture
    body = extract_body(request)
    _span_processor.store_body(span_id, request_body=body)
```

---

## Git Standards

### Commit Messages

**Format:** `<type>: <description>`

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation changes
- `refactor` - Code refactoring
- `test` - Test changes
- `chore` - Build/tooling changes

**Examples:**
```
feat: add HITL approval polling with expiration
fix: clear stale verdicts on workflow restart
docs: update README with guardrails examples
refactor: extract span serialization to helper
test: add integration test for BLOCK verdict
chore: update dependencies to latest versions
```

### Branch Naming

**Format:** `<type>/<description>`

**Examples:**
- `feat/approval-expiration`
- `fix/verdict-staleness`
- `docs/api-reference`
- `refactor/span-processor`

---

**Document Version:** 1.0
**Last Updated:** 2026-01-31
