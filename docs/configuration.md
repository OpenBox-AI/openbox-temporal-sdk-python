# OpenBox SDK Configuration

## Required

| Parameter | Type | Description |
|-----------|------|-------------|
| `openbox_url` | `str` | OpenBox Core API URL (HTTPS required for non-localhost) |
| `openbox_api_key` | `str` | API key (`obx_live_*` or `obx_test_*`) |

## Governance

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `governance_timeout` | `float` | `30.0` | API timeout in seconds |
| `governance_policy` | `str` | `"fail_open"` | `"fail_open"` or `"fail_closed"` |

## Event Filtering

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `send_start_event` | `bool` | `True` | Send WorkflowStarted events |
| `send_activity_start_event` | `bool` | `True` | Send ActivityStarted events |
| `skip_workflow_types` | `set` | `None` | Workflow types to skip |
| `skip_activity_types` | `set` | `None` | Activity types to skip |
| `skip_signals` | `set` | `None` | Signal names to skip |

## Human-in-the-Loop

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hitl_enabled` | `bool` | `True` | Enable approval polling for `REQUIRE_APPROVAL` |
| `hitl_poll_interval_ms` | `int` | `5000` | Polling interval in milliseconds for approval status |

## Identity & Signing (AIP DID)

When provided, every request to OpenBox Core is signed locally with Ed25519 and
carries the agent's DID in signed headers — required for `signing_required=true`
agents. `agent_did` and `agent_private_key` are **both-or-neither**.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_did` | `str` | `None` | Agent DID, format `did:aip:<uuid>`. Asserted in the `X-OpenBox-Agent-DID` signed header. |
| `agent_private_key` | `str` | `None` | Base64-encoded raw 32-byte Ed25519 seed. Signs requests locally. Non-repudiation material — never logged, redacted from errors/`__repr__`. |

The private key is loaded once into a key object at init; the raw seed is never
stored, logged, or echoed in errors. The signing module is kept off the Temporal
workflow sandbox import path.

Commonly supplied from env alongside the API key:

```
OPENBOX_AGENT_DID=did:aip:<uuid>
OPENBOX_AGENT_PRIVATE_KEY=<base64 raw 32-byte Ed25519 seed>
```

## Governance Context

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | `str` | `None` | Session identifier for governance context (optional) |
| `agent_name` | `str` | `None` | Agent name for governance context (optional) |
| `tool_type_map` | `dict` | `{}` | Mapping of tool names to types for governance evaluation |

## Instrumentation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instrument_databases` | `bool` | `True` | Capture database queries (includes sqlite3) |
| `db_libraries` | `set` | `None` | `"psycopg2"`, `"asyncpg"`, `"mysql"`, `"pymysql"`, `"pymongo"`, `"redis"`, `"sqlalchemy"`, `"sqlite3"` |
| `instrument_file_io` | `bool` | `False` | Capture file operations |

## Example

```python
worker = create_openbox_worker(
    client=client,
    task_queue="my-queue",
    workflows=[MyWorkflow],
    activities=[my_activity],

    # Required
    openbox_url=os.getenv("OPENBOX_URL"),
    openbox_api_key=os.getenv("OPENBOX_API_KEY"),

    # Identity & signing (both-or-neither)
    agent_did=os.getenv("OPENBOX_AGENT_DID"),
    agent_private_key=os.getenv("OPENBOX_AGENT_PRIVATE_KEY"),

    # Optional
    governance_policy="fail_closed",
    governance_timeout=15.0,
    hitl_enabled=True,
    skip_workflow_types={"InternalWorkflow"},
    instrument_databases=True,
    db_libraries={"psycopg2", "redis"},
    instrument_file_io=False,
)
```
