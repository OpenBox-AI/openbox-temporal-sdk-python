import pytest

from openbox.sandbox.otel_telemetry import (
    GovernedCommandTelemetryBridge,
    GovernedCommandTerminalRecord,
    parse_image_digest,
)


class _CapturedSpan:
    def add_event(self, *args, **kwargs) -> None:
        pass

    def set_attributes(self, *args, **kwargs) -> None:
        pass

    def set_status(self, *args, **kwargs) -> None:
        pass

    def end(self, *args, **kwargs) -> None:
        pass


class _CapturingTracer:
    def __init__(self) -> None:
        self.attributes = None

    def start_span(self, *args, attributes, **kwargs):
        self.attributes = attributes
        return _CapturedSpan()


def test_native_srt_telemetry_does_not_require_an_oci_image_digest() -> None:
    bridge = GovernedCommandTelemetryBridge(
        "native://srt", sandbox_provider="srt"
    )

    assert bridge.image_digest is None
    assert bridge.sandbox_provider == "srt"
    assert parse_image_digest("native://srt") is None


def test_sandbox_span_uses_namespaced_provider_attribute() -> None:
    tracer = _CapturingTracer()
    record = GovernedCommandTerminalRecord(
        workflow_id="workflow-1",
        run_id="run-1",
        activity_id="activity-1",
        attempt=1,
        profile_id="profile-1",
        workflow_type="DemoWorkflow",
        task_queue="demo",
        image_digest=None,
        sandbox_provider="srt",
        parent_span_context=None,
        started_ns=1,
        ended_ns=2,
        phases=(),
        outcome="success",
        disposition="executed_in_sandbox",
        timeout_status="not_observed",
        cleanup_status="deleted",
        directive="continue",
        error_code="none",
        exit_code=0,
        stdout_bytes=0,
        stderr_bytes=0,
        unsafe_stdout=None,
        unsafe_stderr=None,
    )

    GovernedCommandTelemetryBridge._record_span(tracer, record)

    assert tracer.attributes["openbox.sandbox.provider"] == "srt"
    assert "sandbox.provider" not in tracer.attributes


def test_openshell_telemetry_keeps_immutable_image_digest_validation() -> None:
    digest = "sha256:" + "c" * 64
    assert parse_image_digest("registry.invalid/openbox@" + digest) == digest
    with pytest.raises(ValueError, match="immutable image template rejected"):
        parse_image_digest("registry.invalid/openbox:latest")
