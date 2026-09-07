"""Lazy public exports for Activity-side governed-command integration.

The package initializer stays empty of Activity dependencies so importing
``openbox.sandbox.types`` from workflow code loads only that pure types module.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "TemporalSandboxConfig": "openbox.sandbox.adapter",
    # Declared types of TemporalSandboxConfig fields. A caller has to build a
    # profile bundle and may implement a heartbeat sink, so both belong in the
    # public surface; docs/governed-commands.md already imports the bundle
    # from here.
    "TemporalCommandProfileBundle": "openbox.sandbox.profiles",
    "TemporalHeartbeatSink": "openbox.sandbox.adapter",
    "SandboxConfig": "openbox.sandbox.config",
    "TemporalSandboxConfigurationError": "openbox.sandbox.adapter",

    "DecimalArgument": "openbox.sandbox.registry",
    "EnumArgument": "openbox.sandbox.registry",
    "GovernedCommandDefinition": "openbox.sandbox.registry",
    "GovernedCommandRegistry": "openbox.sandbox.registry",
    "GovernedCommandRegistryError": "openbox.sandbox.registry",
    "IdentifierArgument": "openbox.sandbox.registry",
    "IdentifierResultField": "openbox.sandbox.registry",
    "IntegerResultField": "openbox.sandbox.registry",
    "LiteralArgument": "openbox.sandbox.registry",
    "TypedJsonResultSchema": "openbox.sandbox.registry",
    "governed_command_registry": "openbox.sandbox.registry",
    "GovernedCommandActivityResult": "openbox.sandbox.types",
    "GovernedCommandInputError": "openbox.sandbox.types",
    "GovernedCommandRequest": "openbox.sandbox.types",
    "GovernedCommandResultValue": "openbox.sandbox.types",
    "GovernedCommandTypedResult": "openbox.sandbox.types",
    "StructuredCommandArgument": "openbox.sandbox.types",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
