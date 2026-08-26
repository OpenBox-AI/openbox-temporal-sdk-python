"""Immutable typed application command registry for governed commands.

The registry is the only public way applications define governed commands.
It is constructed from plain typed values in code — never from manifest
files, environment selectors, JSON documents, or signed bundles — and it
produces both the Temporal argv-derivation mapping and the dispatcher
admission bundle from the same in-memory definitions, so the two can never
diverge.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from .profiles import (
    TemporalCommandProfileBundle,
    _ArgumentMapping,
    _Profile,
    _ResultField,
    _ResultSchema,
)

if TYPE_CHECKING:
    from openbox_sandbox.dispatcher.profiles import CommandProfileBundle

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_FIELD = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")
_FORBIDDEN_FIELD_PARTS = (
    "argv",
    "command",
    "cmd",
    "code",
    "secret",
    "token",
    "password",
    "credential",
    "private_key",
)
# The registry is process-lifetime configuration. Both derived bundles use one
# fixed validity window so their canonical fingerprints stay deterministic.
_REGISTRY_ISSUED_AT = datetime(2000, 1, 1, tzinfo=UTC)
_REGISTRY_EXPIRES_AT = datetime(9999, 1, 1, tzinfo=UTC)
_REGISTRY_KEY_ID = "typed-registry"


class GovernedCommandRegistryError(ValueError):
    """Raised when a typed command definition cannot be accepted."""

    def __init__(self) -> None:
        super().__init__("governed command registry rejected")


def _require_field_name(name: object) -> str:
    if not isinstance(name, str) or _FIELD.fullmatch(name) is None:
        raise GovernedCommandRegistryError()
    lowered = name.lower()
    if any(part in lowered for part in _FORBIDDEN_FIELD_PARTS):
        raise GovernedCommandRegistryError()
    return name


@dataclass(frozen=True, slots=True)
class LiteralArgument:
    """A fixed argv token that never varies with Workflow input."""

    value: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.value, str)
            or len(self.value.encode()) > 4096
            or "\x00" in self.value
        ):
            raise GovernedCommandRegistryError()


@dataclass(frozen=True, slots=True)
class IdentifierArgument:
    """A caller-supplied bounded identifier token."""

    field: str
    max_bytes: int = 256

    def __post_init__(self) -> None:
        _require_field_name(self.field)
        if (
            isinstance(self.max_bytes, bool)
            or not isinstance(self.max_bytes, int)
            or not 1 <= self.max_bytes <= 4096
        ):
            raise GovernedCommandRegistryError()


@dataclass(frozen=True, slots=True)
class EnumArgument:
    """A caller-supplied token restricted to a finite fixed choice set."""

    field: str
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_field_name(self.field)
        if (
            not isinstance(self.values, tuple)
            or not self.values
            or len(self.values) > 128
            or len(set(self.values)) != len(self.values)
            or not all(
                isinstance(item, str) and 0 < len(item.encode()) <= 4096 for item in self.values
            )
        ):
            raise GovernedCommandRegistryError()


@dataclass(frozen=True, slots=True)
class DecimalArgument:
    """A caller-supplied bounded base-10 integer token."""

    field: str
    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        _require_field_name(self.field)
        if (
            isinstance(self.minimum, bool)
            or isinstance(self.maximum, bool)
            or not isinstance(self.minimum, int)
            or not isinstance(self.maximum, int)
            or self.minimum > self.maximum
        ):
            raise GovernedCommandRegistryError()


CommandArgument = LiteralArgument | IdentifierArgument | EnumArgument | DecimalArgument


@dataclass(frozen=True, slots=True)
class IdentifierResultField:
    """A bounded identifier admitted from canonical JSON output."""

    name: str
    max_bytes: int = 256

    def __post_init__(self) -> None:
        _require_field_name(self.name)
        if type(self.max_bytes) is not int or not 1 <= self.max_bytes <= 4096:
            raise GovernedCommandRegistryError()

    def _profile_field(self) -> _ResultField:
        return _ResultField(self.name, "identifier", max_bytes=self.max_bytes)

    def _canonical(self) -> dict[str, Any]:
        return {"name": self.name, "kind": "identifier", "max_bytes": self.max_bytes}


@dataclass(frozen=True, slots=True)
class IntegerResultField:
    """A bounded integer admitted from canonical JSON output."""

    name: str
    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        _require_field_name(self.name)
        if (
            type(self.minimum) is not int
            or type(self.maximum) is not int
            or self.minimum > self.maximum
        ):
            raise GovernedCommandRegistryError()

    def _profile_field(self) -> _ResultField:
        return _ResultField(
            self.name,
            "integer",
            minimum=self.minimum,
            maximum=self.maximum,
        )

    def _canonical(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "integer",
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


ResultField = IdentifierResultField | IntegerResultField


@dataclass(frozen=True, slots=True)
class TypedJsonResultSchema:
    """Canonical bounded JSON output admitted into durable Activity results."""

    name: str
    fields: tuple[ResultField, ...]
    max_bytes: int = 16 * 1024

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or _IDENTIFIER.fullmatch(self.name) is None
            or len(self.name.encode("utf-8")) > 128
            or not isinstance(self.fields, tuple)
            or not self.fields
            or len(self.fields) > 64
            or not all(
                isinstance(field, (IdentifierResultField, IntegerResultField))
                for field in self.fields
            )
            or len({field.name for field in self.fields}) != len(self.fields)
            or type(self.max_bytes) is not int
            or not 1 <= self.max_bytes <= 16 * 1024
        ):
            raise GovernedCommandRegistryError()

    def _profile_schema(self) -> _ResultSchema:
        return _ResultSchema(
            self.name,
            self.max_bytes,
            tuple(field._profile_field() for field in self.fields),
        )

    def _canonical(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_bytes": self.max_bytes,
            "fields": [field._canonical() for field in self.fields],
        }


@dataclass(frozen=True, slots=True)
class GovernedCommandDefinition:
    """One typed governed command: absolute executable plus bounded arguments."""

    command_id: str
    executable: str
    arguments: tuple[CommandArgument, ...] = ()
    result_schema: TypedJsonResultSchema | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.command_id, str)
            or _IDENTIFIER.fullmatch(self.command_id) is None
            or len(self.command_id.encode()) > 128
            or not isinstance(self.executable, str)
            or not self.executable.startswith("/")
            or "\x00" in self.executable
            or len(self.executable.encode()) > 4096
            or not isinstance(self.arguments, tuple)
            or len(self.arguments) > 128
            or (
                self.result_schema is not None
                and not isinstance(self.result_schema, TypedJsonResultSchema)
            )
        ):
            raise GovernedCommandRegistryError()
        fields: list[str] = []
        for argument in self.arguments:
            if not isinstance(
                argument,
                (LiteralArgument, IdentifierArgument, EnumArgument, DecimalArgument),
            ):
                raise GovernedCommandRegistryError()
            if not isinstance(argument, LiteralArgument):
                fields.append(argument.field)
        if len(fields) != len(set(fields)):
            raise GovernedCommandRegistryError()

    def _canonical(self) -> dict[str, Any]:
        arguments: list[dict[str, Any]] = []
        for argument in self.arguments:
            if isinstance(argument, LiteralArgument):
                arguments.append({"kind": "literal", "value": argument.value})
            elif isinstance(argument, IdentifierArgument):
                arguments.append(
                    {
                        "kind": "identifier",
                        "field": argument.field,
                        "max_bytes": argument.max_bytes,
                    }
                )
            elif isinstance(argument, EnumArgument):
                arguments.append(
                    {
                        "kind": "enum",
                        "field": argument.field,
                        "values": list(argument.values),
                    }
                )
            else:
                arguments.append(
                    {
                        "kind": "decimal",
                        "field": argument.field,
                        "minimum": argument.minimum,
                        "maximum": argument.maximum,
                    }
                )
        return {
            "command_id": self.command_id,
            "executable": self.executable,
            "arguments": arguments,
            "result_schema": (
                None if self.result_schema is None else self.result_schema._canonical()
            ),
        }


@dataclass(frozen=True, init=False, repr=False)
class GovernedCommandRegistry:
    """Immutable, canonically fingerprinted set of typed command definitions."""

    fingerprint: str
    _definitions: Mapping[str, GovernedCommandDefinition] = field(compare=False)

    def __init__(self, commands: tuple[GovernedCommandDefinition, ...]) -> None:
        if (
            not isinstance(commands, tuple)
            or not commands
            or len(commands) > 1024
            or not all(isinstance(item, GovernedCommandDefinition) for item in commands)
        ):
            raise GovernedCommandRegistryError()
        definitions: dict[str, GovernedCommandDefinition] = {}
        for command in commands:
            if command.command_id in definitions:
                raise GovernedCommandRegistryError()
            definitions[command.command_id] = command
        canonical = json.dumps(
            {
                "schema": "openbox-governed-command-registry/v1",
                "commands": [definitions[key]._canonical() for key in sorted(definitions)],
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        object.__setattr__(self, "fingerprint", hashlib.sha256(canonical).hexdigest())
        object.__setattr__(self, "_definitions", MappingProxyType(definitions))

    def __repr__(self) -> str:
        return (
            f"GovernedCommandRegistry(commands={len(self._definitions)}, "
            f"fingerprint={self.fingerprint!r})"
        )

    @property
    def command_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._definitions))

    @property
    def bundle_version(self) -> str:
        return f"typed-registry-{self.fingerprint[:16]}"

    def temporal_profile_bundle(self) -> TemporalCommandProfileBundle:
        """Build the argv-derivation mapping used by the Activity wrapper."""
        profiles: dict[str, _Profile] = {}
        for command in self._definitions.values():
            mappings: list[_ArgumentMapping] = []
            for argument in command.arguments:
                if isinstance(argument, LiteralArgument):
                    mappings.append(_ArgumentMapping(kind="literal", literal=argument.value))
                elif isinstance(argument, IdentifierArgument):
                    mappings.append(
                        _ArgumentMapping(
                            kind="field_identifier",
                            field=argument.field,
                            max_bytes=argument.max_bytes,
                        )
                    )
                elif isinstance(argument, EnumArgument):
                    mappings.append(
                        _ArgumentMapping(
                            kind="field_enum",
                            field=argument.field,
                            values=argument.values,
                        )
                    )
                else:
                    mappings.append(
                        _ArgumentMapping(
                            kind="field_decimal",
                            field=argument.field,
                            minimum=argument.minimum,
                            maximum=argument.maximum,
                        )
                    )
            profile_fingerprint = hashlib.sha256(
                json.dumps(
                    command._canonical(),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            profiles[command.command_id] = _Profile(
                command.command_id,
                command.executable,
                tuple(mappings),
                profile_fingerprint,
                (
                    None
                    if command.result_schema is None
                    else command.result_schema._profile_schema()
                ),
            )
        bundle = object.__new__(TemporalCommandProfileBundle)
        object.__setattr__(bundle, "schema_version", 1)
        object.__setattr__(bundle, "bundle_version", self.bundle_version)
        object.__setattr__(bundle, "key_id", _REGISTRY_KEY_ID)
        object.__setattr__(bundle, "issued_at", _REGISTRY_ISSUED_AT)
        object.__setattr__(bundle, "expires_at", _REGISTRY_EXPIRES_AT)
        object.__setattr__(bundle, "fingerprint", self.fingerprint)
        object.__setattr__(bundle, "_profiles", MappingProxyType(profiles))
        return bundle

    def dispatcher_profile_bundle(self) -> CommandProfileBundle:
        """Build the independent dispatcher admission bundle."""
        from openbox_sandbox.dispatcher.profiles import (
            ArgumentRule,
            CommandProfile,
            CommandProfileBundle,
        )

        profiles: dict[str, CommandProfile] = {}
        for command in self._definitions.values():
            rules: list[ArgumentRule] = []
            for argument in command.arguments:
                if isinstance(argument, LiteralArgument):
                    rules.append(ArgumentRule(kind="literal", literal=argument.value))
                elif isinstance(argument, IdentifierArgument):
                    rules.append(ArgumentRule(kind="identifier", max_bytes=argument.max_bytes))
                elif isinstance(argument, EnumArgument):
                    rules.append(ArgumentRule(kind="enum", choices=argument.values))
                else:
                    rules.append(
                        ArgumentRule(
                            kind="decimal",
                            minimum=argument.minimum,
                            maximum=argument.maximum,
                        )
                    )
            profiles[command.command_id] = CommandProfile(
                profile_id=command.command_id,
                executable=command.executable,
                arguments=tuple(rules),
                sensitive=False,
                free_form=False,
            )
        bundle = object.__new__(CommandProfileBundle)
        object.__setattr__(bundle, "schema_version", 1)
        object.__setattr__(bundle, "bundle_version", self.bundle_version)
        object.__setattr__(bundle, "key_id", _REGISTRY_KEY_ID)
        object.__setattr__(bundle, "issued_at", _REGISTRY_ISSUED_AT)
        object.__setattr__(bundle, "expires_at", _REGISTRY_EXPIRES_AT)
        object.__setattr__(bundle, "fingerprint", self.fingerprint)
        object.__setattr__(bundle, "_profiles", MappingProxyType(profiles))
        return bundle


def governed_command_registry(
    *commands: GovernedCommandDefinition,
) -> GovernedCommandRegistry:
    """Build an immutable registry from typed command definitions."""
    return GovernedCommandRegistry(tuple(commands))
