"""Non-workflow composition of one shared runtime for a Temporal Worker."""

from __future__ import annotations

from typing import Any

from openbox_core.client import EvaluationClient
from openbox_core.runtime import OpenBoxRuntime

from .config import GovernanceConfig, get_global_config
from .core_adapter import TemporalFrameworkAdapter, get_core_context_store
from .governance_state import TemporalGovernanceState


def create_temporal_runtime(
    governance: GovernanceConfig,
    *,
    api_url: str,
    api_key: str,
    timeout: float,
    on_api_error: str,
    agent_did: str | None,
    signer: Any,
    core_ca_path: str | None,
    state: TemporalGovernanceState,
) -> OpenBoxRuntime:
    """Build the single live Core client/runtime owned by a Worker or plugin.

    ``state`` is the run-scoped ``TemporalGovernanceState`` shared with the
    interceptors: the adapter records completed-hook stops and HITL
    pending-approval markers into it, and the activity interceptor consumes
    them through the same instance.
    """
    from openbox_core.identity import AgentIdentity

    global_config = get_global_config()
    core_config = global_config.get_core_config(governance)
    core_config.api_url = api_url.rstrip("/")
    core_config.api_key = api_key
    core_config.timeout_seconds = timeout
    core_config.on_api_error = on_api_error
    core_config.core_ca_path = core_ca_path
    core_config.sdk_engine = "temporal"

    # Temporal retains its span correlation, HTTP-body capture, and governed
    # command bridge. Installing generic wrappers in parallel would duplicate
    # process-global instrumentation, so the shared runtime owns decisions and
    # transport while those Temporal-specific wrappers remain local.
    core_config.instrumentation.enabled = False

    identity = (
        AgentIdentity(agent_did, signer) if agent_did and signer is not None else None
    )
    client = EvaluationClient(
        core_config.api_url,
        core_config.api_key,
        timeout_seconds=core_config.timeout_seconds,
        on_api_error=core_config.on_api_error,
        identity=identity,
        sdk_engine="temporal",
    )
    adapter = TemporalFrameworkAdapter(
        state,
        hitl_enabled=governance.hitl_enabled,
        skip_hitl_activity_types=governance.skip_hitl_activity_types,
        context_store=get_core_context_store(),
    )
    return OpenBoxRuntime(
        core_config,
        adapter=adapter,
        client=client,
    )
