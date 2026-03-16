# openbox/file_governance_hooks.py
"""File I/O governance hooks — instruments builtins.open().

Wraps file objects with TracedFile to create OTel spans and send
hook-level governance evaluations for every file operation (open,
read, write, readline, readlines, writelines, close).

- "started" evaluations can block file access before it happens
- "completed" evaluations report what happened (informational)
"""

from __future__ import annotations

import logging
from typing import Optional

from . import hook_governance as _hook_gov

logger = logging.getLogger(__name__)


def _build_file_span_data(
    span,
    file_path: str,
    file_mode: str,
    operation: str,
    stage: str,
    error: Optional[str] = None,
    duration_ms: Optional[float] = None,
    data: Optional[str] = None,
    bytes_read: Optional[int] = None,
    bytes_written: Optional[int] = None,
    lines_count: Optional[int] = None,
    operations: Optional[list] = None,
) -> dict:
    """Build span data dict for a file operation (used by governance hooks).

    attributes: OTel-original only. All custom data at root level.
    """
    import time as _time

    span_id_hex, trace_id_hex, parent_span_id = _hook_gov.extract_span_context(span)
    raw_attrs = getattr(span, 'attributes', None)
    attrs = dict(raw_attrs) if raw_attrs else {}

    span_name = getattr(span, 'name', None) or f"file.{operation}"
    now_ns = _time.time_ns()
    duration_ns = int(duration_ms * 1_000_000) if duration_ms else None
    end_time = now_ns if stage == "completed" else None
    start_time = (now_ns - duration_ns) if duration_ns else now_ns

    result = {
        "span_id": span_id_hex,
        "trace_id": trace_id_hex,
        "parent_span_id": parent_span_id,
        "name": span_name,
        "kind": "INTERNAL",
        "stage": stage,
        "start_time": start_time,
        "end_time": end_time,
        "duration_ns": duration_ns,
        "attributes": attrs,
        "status": {"code": "ERROR" if error else "UNSET", "description": error},
        "events": [],
        # Hook type identification
        "hook_type": "file_operation",
        # File-specific root fields
        "file_path": file_path,
        "file_mode": file_mode,
        "file_operation": operation,
        "error": error,
    }

    # Only include optional fields if they have values
    if data is not None:
        result["data"] = data
    if bytes_read is not None:
        result["bytes_read"] = bytes_read
    if bytes_written is not None:
        result["bytes_written"] = bytes_written
    if lines_count is not None:
        result["lines_count"] = lines_count
    if operations is not None:
        result["operations"] = operations

    return result


def setup_file_io_instrumentation() -> bool:
    """Setup file I/O instrumentation by patching built-in open().

    File operations will be captured as spans with:
    - file.path: File path
    - file.mode: Open mode (r, w, a, etc.)
    - file.operation: read, write, etc.
    - file.bytes: Number of bytes read/written

    Returns:
        True if instrumentation was successful
    """
    import builtins
    from opentelemetry import trace

    # Check if already instrumented
    if hasattr(builtins, '_openbox_original_open'):
        logger.debug("File I/O already instrumented")
        return True

    _original_open = builtins.open
    builtins._openbox_original_open = _original_open  # Store for uninstrumentation
    _tracer = trace.get_tracer("openbox.file_io")

    # Paths to skip (noisy system files)
    _skip_patterns = ('/dev/', '/proc/', '/sys/', '__pycache__', '.pyc', '.pyo', '.so', '.dylib')

    class TracedFile:
        """Wrapper around file object to trace read/write operations.

        Also sends hook-level governance evaluations:
        - "started" on open (can block file access)
        - "completed" on close (reports summary of operations performed)
        """

        def __init__(self, file_obj, file_path: str, mode: str, parent_span):
            self._file = file_obj
            self._file_path = file_path
            self._mode = mode
            self._parent_span = parent_span
            self._bytes_read = 0
            self._bytes_written = 0
            self._operations: list = []  # Track operations for governance payload

        def _evaluate_governance(self, operation: str, stage: str, span=None, **extra):
            """Send governance evaluation for a file operation stage.

            Args:
                operation: File operation name (read, write, close, etc.)
                stage: Governance stage (started, completed)
                span: OTel span for this operation. Falls back to parent span.
                **extra: Additional trigger fields (data, bytes_read, etc.)
            """
            if not _hook_gov.is_configured():
                return
            from .types import GovernanceBlockedError
            active_span = span or self._parent_span
            try:
                span_data = _build_file_span_data(
                    active_span, self._file_path, self._mode, operation, stage,
                    **extra,
                )
                _hook_gov.evaluate_sync(
                    active_span,
                    identifier=self._file_path,
                    span_data=span_data,
                )
            except GovernanceBlockedError:
                raise
            except Exception:
                pass  # fail_open handled inside evaluate_sync

        def read(self, size=-1):
            with _tracer.start_as_current_span("file.read") as span:
                span.set_attribute("file.path", self._file_path)
                span.set_attribute("file.operation", "read")
                self._evaluate_governance("read", "started", span=span)

                data = self._file.read(size)
                bytes_count = len(data) if isinstance(data, (str, bytes)) else 0
                self._bytes_read += bytes_count
                self._operations.append("read")
                span.set_attribute("file.bytes", bytes_count)

                self._evaluate_governance("read", "completed", span=span, data=data, bytes_read=bytes_count)
                return data

        def readline(self):
            with _tracer.start_as_current_span("file.readline") as span:
                span.set_attribute("file.path", self._file_path)
                span.set_attribute("file.operation", "readline")
                self._evaluate_governance("readline", "started", span=span)

                data = self._file.readline()
                bytes_count = len(data) if isinstance(data, (str, bytes)) else 0
                self._bytes_read += bytes_count
                self._operations.append("readline")
                span.set_attribute("file.bytes", bytes_count)

                self._evaluate_governance("readline", "completed", span=span, data=data, bytes_read=bytes_count)
                return data

        def readlines(self):
            with _tracer.start_as_current_span("file.readlines") as span:
                span.set_attribute("file.path", self._file_path)
                span.set_attribute("file.operation", "readlines")
                self._evaluate_governance("readlines", "started", span=span)

                data = self._file.readlines()
                bytes_count = sum(len(line) for line in data) if data else 0
                self._bytes_read += bytes_count
                self._operations.append("readlines")
                span.set_attribute("file.bytes", bytes_count)
                span.set_attribute("file.lines", len(data) if data else 0)

                self._evaluate_governance(
                    "readlines", "completed", span=span,
                    data=data, bytes_read=bytes_count,
                    lines_count=len(data) if data else 0,
                )
                return data

        def write(self, data):
            with _tracer.start_as_current_span("file.write") as span:
                span.set_attribute("file.path", self._file_path)
                span.set_attribute("file.operation", "write")
                self._evaluate_governance("write", "started", span=span)

                bytes_count = len(data) if isinstance(data, (str, bytes)) else 0
                span.set_attribute("file.bytes", bytes_count)
                self._bytes_written += bytes_count
                self._operations.append("write")
                result = self._file.write(data)

                self._evaluate_governance("write", "completed", span=span, data=data, bytes_written=bytes_count)
                return result

        def writelines(self, lines):
            with _tracer.start_as_current_span("file.writelines") as span:
                span.set_attribute("file.path", self._file_path)
                span.set_attribute("file.operation", "writelines")
                self._evaluate_governance("writelines", "started", span=span)

                bytes_count = sum(len(line) for line in lines) if lines else 0
                span.set_attribute("file.bytes", bytes_count)
                span.set_attribute("file.lines", len(lines) if lines else 0)
                self._bytes_written += bytes_count
                self._operations.append("writelines")
                result = self._file.writelines(lines)

                self._evaluate_governance(
                    "writelines", "completed", span=span,
                    data=lines, bytes_written=bytes_count,
                    lines_count=len(lines) if lines else 0,
                )
                return result

        def close(self):
            # Governance "completed" — reports what happened during file lifecycle
            # Use try/finally to ensure file handle and span are always cleaned up
            from .types import GovernanceBlockedError
            gov_error = None
            try:
                self._evaluate_governance(
                    "close", "completed", span=self._parent_span,
                    bytes_read=self._bytes_read,
                    bytes_written=self._bytes_written,
                    operations=self._operations,
                )
            except GovernanceBlockedError as e:
                gov_error = e
            finally:
                if self._parent_span:
                    self._parent_span.set_attribute("file.total_bytes_read", self._bytes_read)
                    self._parent_span.set_attribute("file.total_bytes_written", self._bytes_written)
                    self._parent_span.end()
                self._file.close()
            if gov_error:
                raise gov_error

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            # Don't mask the original exception if close() also raises
            try:
                self.close()
            except Exception:
                if exc_type is None:
                    raise
            return False

        def __iter__(self):
            return iter(self._file)

        def __next__(self):
            return next(self._file)

        def __getattr__(self, name):
            return getattr(self._file, name)

    def traced_open(file, mode='r', *args, **kwargs):
        file_str = str(file)

        # Skip system/noisy paths
        if any(p in file_str for p in _skip_patterns):
            return _original_open(file, mode, *args, **kwargs)

        span = _tracer.start_span("file.open")
        span.set_attribute("file.path", file_str)
        span.set_attribute("file.mode", mode)

        # Governance "started" — can block file access before it happens
        if _hook_gov.is_configured():
            from .types import GovernanceBlockedError
            try:
                open_span_data = _build_file_span_data(
                    span, file_str, mode, "open", "started",
                )
                _hook_gov.evaluate_sync(
                    span,
                    identifier=file_str,
                    span_data=open_span_data,
                )
            except GovernanceBlockedError:
                span.set_attribute("error", True)
                span.set_attribute("governance.blocked", True)
                span.end()
                raise
            except Exception:
                pass  # Non-governance errors are swallowed (fail_open handled inside evaluate_sync)

        try:
            file_obj = _original_open(file, mode, *args, **kwargs)
            return TracedFile(file_obj, file_str, mode, span)
        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            span.end()
            raise

    builtins.open = traced_open
    logger.info("Instrumented: file I/O (builtins.open)")
    return True


def uninstrument_file_io() -> None:
    """Restore original open() function."""
    import builtins
    if hasattr(builtins, '_openbox_original_open'):
        builtins.open = builtins._openbox_original_open
        delattr(builtins, '_openbox_original_open')
        logger.info("Uninstrumented: file I/O")
