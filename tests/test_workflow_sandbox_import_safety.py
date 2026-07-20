"""Workflow-sandbox import safety.

Importing ``openbox.workflow_interceptor`` (and the pure modules it pulls in,
including the base-SDK contracts) must NOT load network/crypto/signing
modules.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

# Modules that must never load on a workflow-sandbox import path.
FORBIDDEN_MODULES = (
    "httpx",
    "cryptography",
    "requests",
    "urllib3",
    "opentelemetry.instrumentation",
    # Base-SDK modules with network/crypto/wall-clock/random behavior:
    "openbox_core.client",
    "openbox_core.identity",
    "openbox_core.approvals",
    "openbox_core.runtime",
    "openbox_core.gate",
    "openbox_core.instrumentation",
    # Temporal-side signing/adapter modules deliberately stay off eager import
    # paths. openbox.client / openbox.hitl lazy-import their heavy dependencies.
    "openbox.request_signing",
    "openbox.core_adapter",
)

SANDBOX_IMPORT_TARGETS = (
    "openbox.workflow_interceptor",
    "openbox.errors",
    "openbox.types",
    "openbox_core.contracts.events",
    "openbox_core.contracts.results",
)

_SNIPPET = """
import importlib, json, sys
importlib.import_module({target!r})
loaded = sorted(
    name for name in sys.modules
    if any(name == f or name.startswith(f + ".") for f in {forbidden!r})
)
print(json.dumps(loaded))
"""


@pytest.mark.parametrize("target", SANDBOX_IMPORT_TARGETS)
def test_sandbox_path_imports_stay_pure(target: str) -> None:
    snippet = _SNIPPET.format(target=target, forbidden=FORBIDDEN_MODULES)
    result = subprocess.run(
        [sys.executable, "-c", snippet], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, f"importing {target} failed:\n{result.stderr}"
    loaded = json.loads(result.stdout)
    assert loaded == [], (
        f"importing {target} loaded forbidden modules: {loaded}. "
        "Workflow sandbox paths must never import network/signing/crypto "
        "modules — route Core calls through the send_governance_event activity."
    )
