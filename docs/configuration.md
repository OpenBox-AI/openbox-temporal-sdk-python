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

## Instrumentation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instrument_databases` | `bool` | `True` | Capture database queries |
| `db_libraries` | `set` | `None` | `"psycopg2"`, `"asyncpg"`, `"mysql"`, `"pymysql"`, `"pymongo"`, `"redis"`, `"sqlalchemy"` |
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
