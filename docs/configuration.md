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

## BLOCK-with-Patch Restart

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_patch_restarts` | `int` | `3` | Maximum Continue-As-New restarts triggered by a BLOCK with `patch`. Minimum value: `1`. Applies uniformly across all governance event origins. When the count would exceed this limit, the workflow fails with a non-retryable `GovernancePatchLimitExceeded` error. |

### Semantics

A governance response carrying `BLOCK` with a valid `patch` (containing `new_input`) will:

1. **Stop the current execution path** with a non-retryable error signal
2. **Not retry** the blocked activity with the same input (Temporal activity retry is bypassed)
3. **Restart the workflow via Continue-As-New** using `patch.new_input` as replacement input
   - If `new_input` is `null`, all arguments of the current run are reused exactly
   - If `new_input` is a non-null value, it is passed as one workflow argument
4. **Keep the same Workflow ID** — the previous run closes as `ContinuedAsNew`; a new Run ID is issued for the restarted execution
5. **Track restarts across the chain** — a global counter (per workflow ID) increments with each restart and is bounded by `max_patch_restarts` to prevent infinite loops

This behavior is **event-agnostic**. The following governance event types all support BLOCK with patch:

- `WorkflowStarted`
- `WorkflowCompleted`
- `WorkflowFailed`
- `SignalReceived`
- `ActivityStarted`
- `ActivityCompleted`
- `Handoff`
- Started and completed hooks
- Approval polling responses

**Origins that do NOT restart:** `HALT`, expired approvals, malformed patches, and plain BLOCK (without a valid `patch`) use existing behavior.

### Critical: Idempotency and Side Effects

**This is the most important user-facing hazard.** A patch may be produced *after* side effects have already occurred:

- A policy may return a patch in response to `ActivityCompleted` — after your activity already ran and may have modified external state
- A completed hook may return a patch — after the operation has occurred
- `WorkflowCompleted` or `WorkflowFailed` may return a patch — after your workflow has executed and potentially caused side effects

**You are responsible for ensuring safety:**

- **Activities must remain idempotent.** If an activity is executed, rolls back or completes, and then the workflow restarts with corrected input, the activity must be safe to re-execute (or be designed to avoid re-execution for the same business key).
- **External writes should use stable business idempotency keys**, not the Temporal Run ID (which changes across Continue-As-New). For example, if writing to an external system, use your application's stable resource identifier, not `workflow.get_info().run_id`.
- **Policy authors and governance integrators own emitting safe replacement input.** The SDK does not validate that `new_input` is safe; it assumes the governance policy has vetted the input and the consequences of re-execution.
- **The SDK adds no source-specific restrictions.** You have full control and full responsibility.

Log any external state changes with identifiers your team can use to trace and reconcile restarts. Monitor governance restart counts to detect policy bugs that cause repeated restarts.

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

## Identity & Signing (Okta AI Agent, v2)

An alternative to OpenBox DID: an Okta-synchronized AI Agent identity, signed
with RS256 JWT assertions instead of Ed25519 headers. Requests route to
Core's `/api/v2/*` endpoints instead of `/api/v1/*`. Mutually exclusive with
`agent_did`/`agent_private_key` — configure exactly one identity verification
method. All eight fields below are **all-or-nothing together**.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `openbox_agent_id` | `str` | `None` | OpenBox agent UUID. |
| `organization_id` | `str` | `None` | OpenBox organization ID. |
| `deployment_id` | `str` | `None` | OpenBox deployment ID. |
| `okta_agent_id` | `str` | `None` | External Okta AI Agent ID (asserted as `iss`/`sub`). |
| `okta_agent_key_id` | `str` | `None` | Okta credential `kid` (JWT key-selection identifier). |
| `okta_agent_private_key` | `str` | `None` | PKCS8 PEM RSA private key (2048-bit minimum). Signs requests locally. Non-repudiation material — never logged, redacted from errors/`__repr__`. |
| `okta_agent_algorithm` | `str` | `"RS256"` | Signing algorithm. Only `"RS256"` is supported at launch. |
| `agent_proof_audience` | `str` | `None` | Deployment-specific assertion audience, `urn:openbox:<deployment-id>:core`. |

All RSA key loading, the 2048-bit floor, algorithm allowlisting, and RS256/JWT
assertion construction are owned by the base SDK
(`openbox_core.identity_okta`) — this package never re-implements them.

Commonly supplied from env alongside the API key:

```
OPENBOX_AGENT_ID=<openbox-agent-uuid>
OPENBOX_ORGANIZATION_ID=<openbox-organization-id>
OPENBOX_DEPLOYMENT_ID=<openbox-deployment-id>
OPENBOX_OKTA_AGENT_ID=<external-okta-ai-agent-id>
OPENBOX_OKTA_AGENT_KEY_ID=<kid>
OPENBOX_OKTA_AGENT_PRIVATE_KEY=<pkcs8-pem-rsa-private-key>
OPENBOX_AGENT_PROOF_AUDIENCE=urn:openbox:<deployment-id>:core
```

Note: unlike `agent_did`/`agent_private_key`, these Okta fields are read
directly from the arguments you pass to `create_openbox_worker(...)` /
`OpenBoxPlugin(...)` / `initialize(...)` — this package does not read
`OPENBOX_*` environment variables itself (see the "Standard Worker options"
example in the README for the `os.getenv(...)` pattern this project uses
throughout).

## Governance Context

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | `str` | `None` | Session identifier for governance context (optional) |
| `agent_name` | `str` | `None` | Agent name for governance context (optional) |
| `tool_type_map` | `dict` | `{}` | Mapping of tool names to types for governance evaluation |

## Multi-Agent Sessions

No worker/plugin parameters. Your application supplies the shared
`multi_agent_session_id` per workflow via the Temporal **workflow memo**; the SDK
only propagates it. The SDK never invents a session id.

| Mechanism | Key | Set by |
|-----------|-----|--------|
| Workflow memo | `openbox_multi_agent_session_id` | App, at `start_workflow(memo={...})` |

When the memo is present, the id is attached to every workflow, activity, and
hook governance event, and is propagated from the workflow to its activities
internally. Emit an explicit handoff from workflow code with
`emit_handoff(multi_agent_session_id=..., from_agent_did=...)`. See the README
"Multi-Agent Sessions" section for usage.

## Instrumentation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instrument_databases` | `bool` | `True` | Capture database queries (the base runtime installs every available DB instrumentor, including sqlite3) |
| `db_libraries` | `set` | `None` | Accepted for backward compatibility; no effect. The base runtime installs all available DB instrumentors regardless. |
| `sqlalchemy_engine` | `Any` | `None` | Accepted for backward compatibility; no effect. SQLAlchemy is governed globally via an `Engine` event listener, so pre-existing engines are covered automatically. |
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
    instrument_file_io=False,
)
```
