"""Strict, callable-free Temporal command profile mapping."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

from .types import (
    GovernedCommandInputError,
    GovernedCommandRequest,
    GovernedCommandResultValue,
    GovernedCommandTypedResult,
)

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_FIELD = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_MAX_DOCUMENT = 1024 * 1024
_MAX_RESULT_BODY = 16 * 1024


class CommandProfileBundleError(ValueError):
    def __init__(self) -> None:
        super().__init__("Temporal command profile bundle rejected")


class CommandResultValidationError(ValueError):
    def __init__(self) -> None:
        super().__init__("governed command typed result rejected")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CommandProfileBundleError()
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise CommandProfileBundleError()


def _strict_result_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CommandResultValidationError()
        result[key] = value
    return result


def _reject_result_constant(_: str) -> None:
    raise CommandResultValidationError()


def _object(value: object, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise CommandProfileBundleError()
    return value


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    except (TypeError, ValueError):
        raise CommandProfileBundleError() from None


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise CommandProfileBundleError()
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise CommandProfileBundleError() from None
    return parsed.astimezone(UTC)


@dataclass(frozen=True)
class _ArgumentMapping:
    kind: str
    field: str | None = None
    literal: str | None = None
    values: tuple[str, ...] = ()
    minimum: int | None = None
    maximum: int | None = None
    max_bytes: int | None = None

    @classmethod
    def parse(cls, value: object) -> _ArgumentMapping:
        if not isinstance(value, dict) or not isinstance(value.get("kind"), str):
            raise CommandProfileBundleError()
        kind = value["kind"]
        if kind == "literal":
            item = _object(value, {"kind", "value"})["value"]
            if not isinstance(item, str) or len(item.encode()) > 4096:
                raise CommandProfileBundleError()
            return cls(kind=kind, literal=item)
        if kind == "field_identifier":
            item = _object(value, {"kind", "field", "max_bytes"})
            _field(item["field"])
            maximum = item["max_bytes"]
            if (
                isinstance(maximum, bool)
                or not isinstance(maximum, int)
                or not 1 <= maximum <= 4096
            ):
                raise CommandProfileBundleError()
            return cls(kind=kind, field=item["field"], max_bytes=maximum)
        if kind == "field_enum":
            item = _object(value, {"kind", "field", "values"})
            _field(item["field"])
            values = item["values"]
            if (
                not isinstance(values, list)
                or not values
                or not all(isinstance(choice, str) for choice in values)
                or len(values) != len(set(values))
                or len(values) > 128
            ):
                raise CommandProfileBundleError()
            return cls(kind=kind, field=item["field"], values=tuple(values))
        if kind == "field_decimal":
            item = _object(value, {"kind", "field", "minimum", "maximum"})
            _field(item["field"])
            minimum, maximum = item["minimum"], item["maximum"]
            if (
                isinstance(minimum, bool)
                or isinstance(maximum, bool)
                or not isinstance(minimum, int)
                or not isinstance(maximum, int)
                or minimum > maximum
            ):
                raise CommandProfileBundleError()
            return cls(
                kind=kind,
                field=item["field"],
                minimum=minimum,
                maximum=maximum,
            )
        raise CommandProfileBundleError()

    def map(self, values: Mapping[str, str | int]) -> str:
        if self.kind == "literal":
            assert self.literal is not None
            return self.literal
        assert self.field is not None
        if self.field not in values:
            raise GovernedCommandInputError("governed command input rejected")
        value = values[self.field]
        if self.kind == "field_identifier":
            if (
                not isinstance(value, str)
                or _IDENTIFIER.fullmatch(value) is None
                or len(value.encode()) > self.max_bytes  # type: ignore[operator]
            ):
                raise GovernedCommandInputError("governed command input rejected")
            return value
        if self.kind == "field_enum":
            if not isinstance(value, str) or value not in self.values:
                raise GovernedCommandInputError("governed command input rejected")
            return value
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < self.minimum  # type: ignore[operator]
            or value > self.maximum  # type: ignore[operator]
        ):
            raise GovernedCommandInputError("governed command input rejected")
        return str(value)


def _field(value: object) -> None:
    if not isinstance(value, str) or _FIELD.fullmatch(value) is None:
        raise CommandProfileBundleError()
    lowered = value.lower()
    if any(
        part in lowered
        for part in (
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
    ):
        raise CommandProfileBundleError()


@dataclass(frozen=True)
class _ResultField:
    name: str
    kind: str
    minimum: int | None = None
    maximum: int | None = None
    max_bytes: int | None = None

    @classmethod
    def parse(cls, value: object) -> _ResultField:
        if not isinstance(value, dict) or not isinstance(value.get("kind"), str):
            raise CommandProfileBundleError()
        kind = value["kind"]
        if kind == "identifier":
            item = _object(value, {"name", "kind", "max_bytes"})
            _field(item["name"])
            maximum = item["max_bytes"]
            if type(maximum) is not int or not 1 <= maximum <= 4096:
                raise CommandProfileBundleError()
            return cls(item["name"], kind, max_bytes=maximum)
        if kind == "integer":
            item = _object(value, {"name", "kind", "minimum", "maximum"})
            _field(item["name"])
            minimum, maximum = item["minimum"], item["maximum"]
            if (
                type(minimum) is not int
                or type(maximum) is not int
                or minimum > maximum
            ):
                raise CommandProfileBundleError()
            return cls(item["name"], kind, minimum=minimum, maximum=maximum)
        raise CommandProfileBundleError()

    def validate(self, value: object) -> GovernedCommandResultValue:
        if self.kind == "identifier":
            if (
                not isinstance(value, str)
                or _IDENTIFIER.fullmatch(value) is None
                or len(value.encode("utf-8")) > self.max_bytes  # type: ignore[operator]
            ):
                raise CommandResultValidationError()
        elif (
            type(value) is not int
            or value < self.minimum  # type: ignore[operator]
            or value > self.maximum  # type: ignore[operator]
        ):
            raise CommandResultValidationError()
        try:
            return GovernedCommandResultValue(self.name, value)  # type: ignore[arg-type]
        except GovernedCommandInputError:
            raise CommandResultValidationError() from None


@dataclass(frozen=True)
class _ResultSchema:
    name: str
    max_bytes: int
    fields: tuple[_ResultField, ...]

    @classmethod
    def parse(cls, value: object) -> _ResultSchema:
        item = _object(value, {"name", "max_bytes", "fields"})
        name, maximum, raw_fields = item["name"], item["max_bytes"], item["fields"]
        if (
            not isinstance(name, str)
            or _IDENTIFIER.fullmatch(name) is None
            or len(name.encode("utf-8")) > 128
            or type(maximum) is not int
            or not 1 <= maximum <= _MAX_RESULT_BODY
            or not isinstance(raw_fields, list)
            or not raw_fields
            or len(raw_fields) > 64
        ):
            raise CommandProfileBundleError()
        fields = tuple(_ResultField.parse(field) for field in raw_fields)
        names = [field.name for field in fields]
        if len(names) != len(set(names)):
            raise CommandProfileBundleError()
        return cls(name, maximum, fields)

    def parse_output(self, output: bytes) -> GovernedCommandTypedResult:
        if type(output) is not bytes or not output or len(output) > self.max_bytes:
            raise CommandResultValidationError()
        try:
            text = output.decode("utf-8")
            value = json.loads(
                text,
                object_pairs_hook=_strict_result_object,
                parse_constant=_reject_result_constant,
            )
        except ValueError:
            raise CommandResultValidationError() from None
        if not isinstance(value, dict) or set(value) != {
            field.name for field in self.fields
        }:
            raise CommandResultValidationError()
        try:
            canonical = json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError):
            raise CommandResultValidationError() from None
        if output != canonical:
            raise CommandResultValidationError()
        return GovernedCommandTypedResult(
            self.name,
            tuple(field.validate(value[field.name]) for field in self.fields),
        )


@dataclass(frozen=True)
class _Profile:
    profile_id: str
    executable: str
    arguments: tuple[_ArgumentMapping, ...]
    fingerprint: str
    result_schema: _ResultSchema | None = None

    def derive(self, request: GovernedCommandRequest) -> tuple[str, ...]:
        values = {item.name: item.value for item in request.arguments}
        expected = {item.field for item in self.arguments if item.field is not None}
        if set(values) != expected:
            raise GovernedCommandInputError("governed command input rejected")
        return (self.executable, *(item.map(values) for item in self.arguments))

    def parse_result(self, output: bytes) -> GovernedCommandTypedResult | None:
        if self.result_schema is None:
            return None
        return self.result_schema.parse_output(output)


def _parse_profile_values(value: object) -> dict[str, _Profile]:
    if not isinstance(value, list) or not value or len(value) > 1024:
        raise CommandProfileBundleError()
    profiles: dict[str, _Profile] = {}
    base_profile_fields = {
        "id",
        "executable",
        "arguments",
        "sensitive",
        "free_form",
        "result_mode",
    }
    for raw in value:
        if not isinstance(raw, dict):
            raise CommandProfileBundleError()
        if (
            set(raw) == base_profile_fields
            and raw.get("result_mode") == "metadata_only"
        ):
            item = raw
            result_schema = None
        elif (
            set(raw) == base_profile_fields | {"result_schema"}
            and raw.get("result_mode") == "typed_json_v1"
        ):
            item = raw
            result_schema = _ResultSchema.parse(raw["result_schema"])
        else:
            raise CommandProfileBundleError()
        profile_id, executable, arguments = (
            item["id"],
            item["executable"],
            item["arguments"],
        )
        if (
            not isinstance(profile_id, str)
            or _IDENTIFIER.fullmatch(profile_id) is None
            or len(profile_id.encode()) > 128
            or profile_id in profiles
            or not isinstance(executable, str)
            or not executable.startswith("/")
            or "\x00" in executable
            or len(executable.encode()) > 4096
            or not isinstance(arguments, list)
            or len(arguments) > 128
            or item["sensitive"] is not False
            or item["free_form"] is not False
        ):
            raise CommandProfileBundleError()
        mappings = tuple(_ArgumentMapping.parse(item) for item in arguments)
        fields = [item.field for item in mappings if item.field is not None]
        if len(fields) != len(set(fields)):
            raise CommandProfileBundleError()
        profiles[profile_id] = _Profile(
            profile_id,
            executable,
            mappings,
            hashlib.sha256(_canonical(item)).hexdigest(),
            result_schema,
        )
    return profiles


@dataclass(frozen=True, init=False)
class TemporalCommandProfileBundle:
    schema_version: int
    bundle_version: str
    key_id: str
    issued_at: datetime
    expires_at: datetime
    fingerprint: str
    _profiles: Mapping[str, _Profile]

    def __init__(self) -> None:
        raise TypeError("use load() or from_trusted() to construct Temporal profiles")

    @classmethod
    def from_trusted(
        cls,
        *,
        bundle_version: str,
        issued_at: datetime,
        expires_at: datetime,
        profiles: Sequence[Mapping[str, Any]],
        now: datetime,
    ) -> TemporalCommandProfileBundle:
        """Build immutable mappings from profiles owned by this process."""
        return _trusted_bundle(
            cls,
            bundle_version=bundle_version,
            issued_at=issued_at,
            expires_at=expires_at,
            profiles=profiles,
            now=now,
        )

    @classmethod
    def load(
        cls,
        document: bytes | str,
        *,
        secret: bytes,
        expected_key_id: str,
        now: datetime | None = None,
    ) -> TemporalCommandProfileBundle:
        if not isinstance(secret, bytes) or len(secret) < 32 or not expected_key_id:
            raise CommandProfileBundleError()
        body = document.encode() if isinstance(document, str) else document
        if not isinstance(body, bytes) or not body or len(body) > _MAX_DOCUMENT:
            raise CommandProfileBundleError()
        try:
            root = json.loads(
                body,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_constant,
            )
        except (
            json.JSONDecodeError,
            UnicodeDecodeError,
            CommandProfileBundleError,
        ):
            raise CommandProfileBundleError() from None
        root = _object(root, {"payload", "signature"})
        payload = _object(
            root["payload"],
            {
                "schema_version",
                "bundle_version",
                "key_id",
                "issued_at",
                "expires_at",
                "profiles",
            },
        )
        signature = _object(root["signature"], {"algorithm", "key_id", "value"})
        if (
            signature["algorithm"] != "hmac-sha256"
            or signature["key_id"] != expected_key_id
            or payload["key_id"] != expected_key_id
            or not isinstance(signature["value"], str)
            or _HEX.fullmatch(signature["value"]) is None
            or not hmac.compare_digest(
                signature["value"],
                hmac.new(secret, _canonical(payload), hashlib.sha256).hexdigest(),
            )
        ):
            raise CommandProfileBundleError()
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or not isinstance(payload["bundle_version"], str)
            or not payload["bundle_version"]
        ):
            raise CommandProfileBundleError()
        issued, expires = (
            _timestamp(payload["issued_at"]),
            _timestamp(payload["expires_at"]),
        )
        current = (now or datetime.now(UTC)).astimezone(UTC)
        if issued > current or expires <= current or issued >= expires:
            raise CommandProfileBundleError()
        profile_values = payload["profiles"]
        profiles = _parse_profile_values(profile_values)
        instance = object.__new__(cls)
        object.__setattr__(instance, "schema_version", 1)
        object.__setattr__(instance, "bundle_version", payload["bundle_version"])
        object.__setattr__(instance, "key_id", expected_key_id)
        object.__setattr__(instance, "issued_at", issued)
        object.__setattr__(instance, "expires_at", expires)
        object.__setattr__(
            instance, "fingerprint", hashlib.sha256(_canonical(payload)).hexdigest()
        )
        object.__setattr__(instance, "_profiles", MappingProxyType(profiles))
        return instance

    @property
    def profile_ids(self) -> tuple[str, ...]:
        """Return the validated profile identifiers in stable order."""
        return tuple(sorted(self._profiles))

    def derive(
        self, request: GovernedCommandRequest, *, now: datetime | None = None
    ) -> tuple[str, ...]:
        current = (now or datetime.now(UTC)).astimezone(UTC)
        profile = self._profiles.get(request.profile_id)
        if profile is None or not self.issued_at <= current < self.expires_at:
            raise GovernedCommandInputError("governed command input rejected")
        return profile.derive(request)

    def profile_fingerprint(
        self, profile_id: str, *, now: datetime | None = None
    ) -> str:
        """Return a stable identity for one validated profile definition."""
        current = (now or datetime.now(UTC)).astimezone(UTC)
        profile = self._profiles.get(profile_id)
        if profile is None or not self.issued_at <= current < self.expires_at:
            raise GovernedCommandInputError("governed command input rejected")
        return profile.fingerprint

    def parse_result(
        self,
        profile_id: str,
        output: bytes,
        *,
        now: datetime | None = None,
    ) -> GovernedCommandTypedResult | None:
        """Return only profile-admitted values, never the raw output body."""
        current = (now or datetime.now(UTC)).astimezone(UTC)
        profile = self._profiles.get(profile_id)
        if profile is None or not self.issued_at <= current < self.expires_at:
            raise CommandResultValidationError()
        return profile.parse_result(output)


def _trusted_bundle(
    bundle_type: type[TemporalCommandProfileBundle],
    *,
    bundle_version: str,
    issued_at: datetime,
    expires_at: datetime,
    profiles: Sequence[Mapping[str, Any]],
    now: datetime,
) -> TemporalCommandProfileBundle:
    if (
        not isinstance(bundle_version, str)
        or not bundle_version
        or not isinstance(issued_at, datetime)
        or issued_at.tzinfo is None
        or not isinstance(expires_at, datetime)
        or expires_at.tzinfo is None
        or not isinstance(now, datetime)
        or now.tzinfo is None
        or isinstance(profiles, (str, bytes))
        or not isinstance(profiles, Sequence)
        or not profiles
        or len(profiles) > 1024
    ):
        raise CommandProfileBundleError()
    issued = issued_at.astimezone(UTC)
    expires = expires_at.astimezone(UTC)
    current = now.astimezone(UTC)
    if issued > current or expires <= current or issued >= expires:
        raise CommandProfileBundleError()

    profile_values = list(profiles)
    parsed = _parse_profile_values(profile_values)

    identity = {
        "schema_version": 1,
        "bundle_version": bundle_version,
        "issued_at": issued.isoformat(),
        "expires_at": expires.isoformat(),
        "profiles": profile_values,
    }
    instance = object.__new__(bundle_type)
    object.__setattr__(instance, "schema_version", 1)
    object.__setattr__(instance, "bundle_version", bundle_version)
    object.__setattr__(instance, "key_id", "")
    object.__setattr__(instance, "issued_at", issued)
    object.__setattr__(instance, "expires_at", expires)
    object.__setattr__(
        instance, "fingerprint", hashlib.sha256(_canonical(identity)).hexdigest()
    )
    object.__setattr__(instance, "_profiles", MappingProxyType(parsed))
    return instance
