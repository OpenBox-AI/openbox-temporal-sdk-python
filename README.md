# OpenBox SDK for Temporal Workflows

OpenBox SDK provides **governance and observability** for Temporal workflows by capturing workflow/activity lifecycle events, HTTP telemetry, database queries, and file operations, then sending them to OpenBox Core for policy evaluation.

**Key Features:**
- 6 event types (WorkflowStarted, WorkflowCompleted, WorkflowFailed, SignalReceived, ActivityStarted, ActivityCompleted)
- 5-tier verdict system (ALLOW, CONSTRAIN, REQUIRE_APPROVAL, BLOCK, HALT)
- HTTP/Database/File I/O instrumentation via OpenTelemetry
- Guardrails: Input/output validation and redaction
- Human-in-the-loop approval with expiration handling
- Zero-code setup via `create_openbox_worker()` factory

---

## Installation

```bash
pip install openbox-temporal-sdk-python
```

**Requirements:**
- Python 3.9+
- Temporal SDK 1.8+
- OpenTelemetry API/SDK 1.38.0+

---

## Quick Start

Use the `create_openbox_worker()` factory for simple integration:

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

The factory automatically:
1. Validates the API key
2. Creates span processor
3. Sets up OpenTelemetry instrumentation
4. Creates governance interceptors
5. Adds `send_governance_event` activity
6. Returns fully configured Worker

---

## Configuration

### Environment Variables

```bash
OPENBOX_URL=http://localhost:8086
OPENBOX_API_KEY=obx_test_key_1
OPENBOX_GOVERNANCE_TIMEOUT=30.0
OPENBOX_GOVERNANCE_POLICY=fail_open  # or fail_closed
```

### Factory Function Parameters

```python
worker = create_openbox_worker(
    client=client,
    task_queue="my-task-queue",
    workflows=[MyWorkflow],
    activities=[my_activity],

    # OpenBox config
    openbox_url="http://localhost:8086",
    openbox_api_key="obx_test_key_1",
    governance_timeout=30.0,
    governance_policy="fail_open",

    # Event filtering
    send_start_event=True,
    send_activity_start_event=True,
    skip_workflow_types={"InternalWorkflow"},
    skip_activity_types={"send_governance_event"},
    skip_signals={"heartbeat"},

    # Database instrumentation
    instrument_databases=True,
    db_libraries={"psycopg2", "sqlalchemy"},  # None = all available
    sqlalchemy_engine=engine,  # pass pre-existing engine for query capture

    # File I/O instrumentation
    instrument_file_io=False,  # disabled by default

    # Standard Worker options (all supported)
    activity_executor=my_executor,
    max_concurrent_activities=10,
)
```

---

## Governance Verdicts

OpenBox Core returns a verdict indicating what action the SDK should take.

| Verdict | Behavior |
|---------|----------|
| `ALLOW` | Continue execution normally |
| `CONSTRAIN` | Log constraints, continue |
| `REQUIRE_APPROVAL` | Pause, poll for human approval |
| `BLOCK` | Raise error, stop activity |
| `HALT` | Raise error, terminate workflow |

**v1.0 Backward Compatibility:**
- `"continue"` → `ALLOW`
- `"stop"` → `HALT`
- `"require-approval"` → `REQUIRE_APPROVAL`

---

## Event Types

| Event | Trigger | Captured Fields |
|-------|---------|-----------------|
| WorkflowStarted | Workflow begins | workflow_id, run_id, workflow_type, task_queue |
| WorkflowCompleted | Workflow succeeds | workflow_id, run_id, workflow_type |
| WorkflowFailed | Workflow fails | workflow_id, run_id, workflow_type, error |
| SignalReceived | Signal received | workflow_id, signal_name, signal_args |
| ActivityStarted | Activity begins | activity_id, activity_type, activity_input |
| ActivityCompleted | Activity ends | activity_id, activity_type, activity_input, activity_output, spans, status, duration |

---

## Guardrails (Input/Output Redaction)

OpenBox Core can validate and redact sensitive data before/after activity execution:

```python
# Request
{
  "verdict": "allow",
  "guardrails_result": {
    "input_type": "activity_input",
    "redacted_input": {"prompt": "[REDACTED]", "user_id": "123"},
    "validation_passed": true,
    "reasons": []
  }
}

# If validation fails:
{
  "validation_passed": false,
  "reasons": [
    {"type": "pii", "field": "email", "reason": "Contains PII"}
  ]
}
```

---

## Error Handling

Configure error policy via `on_api_error`:

| Policy | Behavior |
|--------|----------|
| `fail_open` (default) | If governance API fails, allow workflow to continue |
| `fail_closed` | If governance API fails, terminate workflow |

---

## Supported Instrumentation

### HTTP Libraries
- `httpx` (sync + async) - full body capture
- `requests` - full body capture
- `urllib3` - full body capture
- `urllib` - request body only

### Databases
- PostgreSQL: `psycopg2`, `asyncpg`
- MySQL: `mysql-connector-python`, `pymysql`
- MongoDB: `pymongo`
- Redis: `redis`
- ORM: `sqlalchemy`

**SQLAlchemy Note:** If your SQLAlchemy engine is created before `create_openbox_worker()` runs (e.g., at module import time), you must pass it via the `sqlalchemy_engine` parameter. Without this, `SQLAlchemyInstrumentor` only patches future `create_engine()` calls and won't capture queries on pre-existing engines.

```python
from db.engine import engine

worker = create_openbox_worker(
    ...,
    db_libraries={"psycopg2", "sqlalchemy"},
    sqlalchemy_engine=engine,
)
```

### File I/O
- `open()`, `read()`, `write()`, `readline()`, `readlines()`
- Skips system paths (`/dev/`, `/proc/`, `/sys/`, `__pycache__`)

---

## Architecture

See [System Architecture](./docs/system-architecture.md) for detailed component design.

**High-Level Flow:**

```
Workflow/Activity → Interceptors → Span Processor → OpenBox Core API
                                                    ↓
                                            Returns Verdict
                                                    ↓
                                    (ALLOW, BLOCK, HALT, etc.)
```

---

## Advanced Usage

For manual control, import individual components:

```python
from openbox import (
    initialize,
    WorkflowSpanProcessor,
    GovernanceInterceptor,
    GovernanceConfig,
)
from openbox.otel_setup import setup_opentelemetry_for_governance
from openbox.activity_interceptor import ActivityGovernanceInterceptor
from openbox.activities import send_governance_event

# 1. Initialize SDK
initialize(api_url="http://localhost:8086", api_key="obx_test_key_1")

# 2. Create span processor
span_processor = WorkflowSpanProcessor(
    ignored_url_prefixes=["http://localhost:8086"]
)

# 3. Setup OTel instrumentation
setup_opentelemetry_for_governance(
    span_processor,
    sqlalchemy_engine=engine,  # optional: instrument pre-existing engine
)

# 4. Create governance config
config = GovernanceConfig(
    on_api_error="fail_closed",
    api_timeout=30.0,
)

# 5. Create interceptors
workflow_interceptor = GovernanceInterceptor(
    api_url="http://localhost:8086",
    api_key="obx_test_key_1",
    span_processor=span_processor,
    config=config,
)

activity_interceptor = ActivityGovernanceInterceptor(
    api_url="http://localhost:8086",
    api_key="obx_test_key_1",
    span_processor=span_processor,
    config=config,
)

# 6. Create worker
from temporalio.worker import Worker
worker = Worker(
    client=client,
    task_queue="my-task-queue",
    workflows=[MyWorkflow],
    activities=[my_activity, send_governance_event],
    interceptors=[workflow_interceptor, activity_interceptor],
)
```

---

## Documentation

- **[Project Overview & PDR](./docs/project-overview-pdr.md)** - Requirements, features, constraints
- **[System Architecture](./docs/system-architecture.md)** - Component design, data flows, security
- **[Codebase Summary](./docs/codebase-summary.md)** - Code structure and component details
- **[Code Standards](./docs/code-standards.md)** - Coding conventions and best practices
- **[Project Roadmap](./docs/project-roadmap.md)** - Future enhancements and timeline

---

## Testing

The SDK includes comprehensive test coverage with 10 test files:

```bash
pytest tests/
```

Test files: `test_activities.py`, `test_activity_interceptor.py`, `test_config.py`, `test_otel_setup.py`, `test_span_processor.py`, `test_tracing.py`, `test_types.py`, `test_worker.py`, `test_workflow_interceptor.py`

---

## License

MIT License - See LICENSE file for details

---

## Support

- **Issues:** GitHub Issues
- **Documentation:** See `./docs/`

---

**Version:** 1.0.1 | **Last Updated:** 2026-02-12
