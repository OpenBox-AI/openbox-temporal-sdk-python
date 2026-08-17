import pytest

from openbox.sandbox.otel_telemetry import (
    GovernedCommandTelemetryBridge,
    parse_image_digest,
)


def test_native_srt_telemetry_does_not_require_an_oci_image_digest() -> None:
    bridge = GovernedCommandTelemetryBridge(
        "native://srt", sandbox_provider="srt"
    )

    assert bridge.image_digest is None
    assert bridge.sandbox_provider == "srt"
    assert parse_image_digest("native://srt") is None


def test_openshell_telemetry_keeps_immutable_image_digest_validation() -> None:
    digest = "sha256:" + "c" * 64
    assert parse_image_digest("registry.invalid/openbox@" + digest) == digest
    with pytest.raises(ValueError, match="immutable image template rejected"):
        parse_image_digest("registry.invalid/openbox:latest")
