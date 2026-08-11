"""Sandbox-command Activity types shared with deterministic Workflow code."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

GOVERNED_COMMAND_ACTIVITY_TYPE = "openbox_governed_command"
_MAX_ARGUMENTS = 64
_MAX_RESULT_FIELDS = 64
_MAX_VALUE_BYTES = 4096
_FIELD_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")
_PROFILE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_FORBIDDEN_NAMES = {
    "argv",
    "command",
    "cmd",
    "code",
    "password",
    "secret",
    "token",
    "credential",
    "private_key",
}


class GovernedCommandInputError(ValueError):
    """Raised before scheduling when structured command input is unsafe."""


@dataclass(frozen=True)
class StructuredCommandArgument:
    name: str
    value: str | int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or _FIELD_NAME.fullmatch(self.name) is None
            or any(part in self.name.lower() for part in _FORBIDDEN_NAMES)
            or isinstance(self.value, bool)
            or not isinstance(self.value, (str, int))
            or len(str(self.value).encode("utf-8")) > _MAX_VALUE_BYTES
        ):
            raise GovernedCommandInputError("governed command input rejected")


@dataclass(frozen=True, init=False)
class GovernedCommandRequest:
    profile_id: str
    arguments: tuple[StructuredCommandArgument, ...]
    governance: dict[str, Any] | None = None

    def __init__(
        self,
        profile_id: str,
        arguments: (
            Mapping[str, str | int]
            | tuple[StructuredCommandArgument, ...]
            | list[StructuredCommandArgument]
        ),
        governance: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(profile_id, str) or _PROFILE_ID.fullmatch(profile_id) is None:
            raise GovernedCommandInputError("governed command input rejected")
        if isinstance(arguments, Mapping):
            snapshot = tuple(
                StructuredCommandArgument(name, value)
                for name, value in arguments.items()
            )
        elif isinstance(arguments, (tuple, list)):
            snapshot = tuple(arguments)
            if not all(
                isinstance(item, StructuredCommandArgument) for item in snapshot
            ):
                raise GovernedCommandInputError("governed command input rejected")
        else:
            raise GovernedCommandInputError("governed command input rejected")
        names = [item.name for item in snapshot]
        if len(snapshot) > _MAX_ARGUMENTS or len(set(names)) != len(names):
            raise GovernedCommandInputError("governed command input rejected")
        if governance is not None and (
            not isinstance(governance, dict)
            or not isinstance(governance.get("verdict"), str)
            or not governance["verdict"]
        ):
            raise GovernedCommandInputError("governed command governance rejected")
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "arguments", snapshot)
        object.__setattr__(self, "governance", governance)

    def to_history_value(self) -> dict[str, Any]:
        """Return a bounded wire value."""
        value: dict[str, Any] = {
            "profile_id": self.profile_id,
            "arguments": [
                {"name": item.name, "value": item.value} for item in self.arguments
            ],
        }
        if self.governance is not None:
            value["governance"] = self.governance
        return value

    @classmethod
    def from_value(cls, value: Any) -> GovernedCommandRequest:
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict) or set(value) not in (
            {"profile_id", "arguments"},
            {"profile_id", "arguments", "governance"},
        ):
            raise GovernedCommandInputError("governed command input rejected")
        arguments = value["arguments"]
        governance = value.get("governance")
        if isinstance(arguments, dict):
            return cls(value["profile_id"], arguments, governance)
        if isinstance(arguments, list):
            converted: list[StructuredCommandArgument] = []
            for item in arguments:
                if not isinstance(item, dict) or set(item) != {"name", "value"}:
                    raise GovernedCommandInputError("governed command input rejected")
                converted.append(StructuredCommandArgument(item["name"], item["value"]))
            return cls(value["profile_id"], converted, governance)
        raise GovernedCommandInputError("governed command input rejected")


@dataclass(frozen=True)
class GovernedCommandResultValue:
    """One schema-validated value; never a raw command-output body."""

    name: str
    value: str | int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or _FIELD_NAME.fullmatch(self.name) is None
            or any(part in self.name.lower() for part in _FORBIDDEN_NAMES)
            or isinstance(self.value, bool)
            or not isinstance(self.value, (str, int))
            or (
                isinstance(self.value, str)
                and len(self.value.encode("utf-8")) > _MAX_VALUE_BYTES
            )
        ):
            raise GovernedCommandInputError("governed command result rejected")


@dataclass(frozen=True)
class GovernedCommandTypedResult:
    """Named, ordered result values admitted by an authenticated profile schema."""

    schema_name: str
    values: tuple[GovernedCommandResultValue, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema_name, str)
            or _PROFILE_ID.fullmatch(self.schema_name) is None
            or not isinstance(self.values, tuple)
            or not self.values
            or len(self.values) > _MAX_RESULT_FIELDS
            or not all(
                isinstance(item, GovernedCommandResultValue) for item in self.values
            )
        ):
            raise GovernedCommandInputError("governed command result rejected")
        names = [item.name for item in self.values]
        if len(names) != len(set(names)):
            raise GovernedCommandInputError("governed command result rejected")


@dataclass(frozen=True)
class GovernedCommandActivityResult:
    profile_id: str
    disposition: str
    exit_code: int
    timeout_status: str
    cleanup_status: str
    stdout_bytes: int
    stderr_bytes: int
    typed_result: GovernedCommandTypedResult | None = None
