# tests/test_tracing.py
"""
Comprehensive tests for openbox/tracing.py module.

Tests cover:
- _safe_serialize() function
- _is_async_function() function
- _set_args_attributes() function
- @traced decorator (sync and async)
- create_span() function
"""

import asyncio
import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

import openbox.hook_governance as hook_gov
from openbox.tracing import (
    _safe_serialize,
    _is_async_function,
    _set_args_attributes,
    traced,
    create_span,
    _get_tracer,
)
from openbox.types import GovernanceBlockedError, WorkflowSpanBuffer


# =============================================================================
# Tests for _safe_serialize()
# =============================================================================


class TestSafeSerialize:
    """Tests for the _safe_serialize function."""

    def test_none_returns_null(self):
        """Test that None is serialized as 'null'."""
        assert _safe_serialize(None) == "null"

    def test_string_serialization(self):
        """Test string values are serialized correctly."""
        assert _safe_serialize("hello") == "hello"
        assert _safe_serialize("") == ""
        assert _safe_serialize("test with spaces") == "test with spaces"

    def test_int_serialization(self):
        """Test integer values are serialized correctly."""
        assert _safe_serialize(0) == "0"
        assert _safe_serialize(42) == "42"
        assert _safe_serialize(-100) == "-100"
        assert _safe_serialize(999999999) == "999999999"

    def test_float_serialization(self):
        """Test float values are serialized correctly."""
        assert _safe_serialize(3.14) == "3.14"
        assert _safe_serialize(0.0) == "0.0"
        assert _safe_serialize(-2.5) == "-2.5"

    def test_bool_serialization(self):
        """Test boolean values are serialized correctly."""
        assert _safe_serialize(True) == "True"
        assert _safe_serialize(False) == "False"

    def test_list_serialization(self):
        """Test list values are serialized as JSON."""
        assert _safe_serialize([]) == "[]"
        assert _safe_serialize([1, 2, 3]) == "[1, 2, 3]"
        assert _safe_serialize(["a", "b"]) == '["a", "b"]'
        assert _safe_serialize([1, "mixed", True]) == '[1, "mixed", true]'

    def test_dict_serialization(self):
        """Test dict values are serialized as JSON."""
        assert _safe_serialize({}) == "{}"
        assert _safe_serialize({"key": "value"}) == '{"key": "value"}'
        assert _safe_serialize({"num": 42}) == '{"num": 42}'
        assert _safe_serialize({"nested": {"a": 1}}) == '{"nested": {"a": 1}}'

    def test_nested_structures(self):
        """Test nested list and dict structures."""
        nested = {"users": [{"name": "Alice"}, {"name": "Bob"}]}
        result = _safe_serialize(nested)
        assert json.loads(result) == nested

    def test_truncation_at_max_length(self):
        """Test that long values are truncated at max_length."""
        long_string = "x" * 3000
        result = _safe_serialize(long_string, max_length=100)
        assert len(result) == 100 + len("...[truncated]")
        assert result.endswith("...[truncated]")
        assert result.startswith("x" * 100)

    def test_truncation_default_max_length(self):
        """Test truncation uses default max_length of 2000."""
        long_string = "y" * 3000
        result = _safe_serialize(long_string)
        assert len(result) == 2000 + len("...[truncated]")
        assert result.endswith("...[truncated]")

    def test_no_truncation_when_under_max_length(self):
        """Test values under max_length are not truncated."""
        short_string = "z" * 50
        result = _safe_serialize(short_string, max_length=100)
        assert result == short_string
        assert "truncated" not in result

    def test_truncation_exact_max_length(self):
        """Test values at exact max_length are not truncated."""
        exact_string = "a" * 100
        result = _safe_serialize(exact_string, max_length=100)
        assert result == exact_string
        assert "truncated" not in result

    def test_unserializable_object_returns_marker(self):
        """Test that unserializable objects return '<unserializable>'."""
        # Create an object that will fail json.dumps with default=str
        class BadRepr:
            def __str__(self):
                raise ValueError("Cannot stringify")
            def __repr__(self):
                raise ValueError("Cannot repr")

        result = _safe_serialize(BadRepr())
        assert result == "<unserializable>"

    def test_custom_object_uses_str(self):
        """Test that custom objects without JSON support use str()."""
        class CustomObj:
            def __str__(self):
                return "CustomObj(value=42)"

        result = _safe_serialize(CustomObj())
        assert result == "CustomObj(value=42)"

    def test_dict_with_non_serializable_values(self):
        """Test dict with non-serializable values uses default=str."""
        class Custom:
            def __str__(self):
                return "custom_value"

        data = {"obj": Custom()}
        result = _safe_serialize(data)
        assert "custom_value" in result


# =============================================================================
# Tests for _is_async_function()
# =============================================================================


class TestIsAsyncFunction:
    """Tests for the _is_async_function helper."""

    def test_identifies_sync_function(self):
        """Test that sync functions are identified as non-async."""
        def sync_func():
            return 42

        assert _is_async_function(sync_func) is False

    def test_identifies_async_function(self):
        """Test that async functions are identified correctly."""
        async def async_func():
            return 42

        assert _is_async_function(async_func) is True

    def test_identifies_lambda_as_sync(self):
        """Test that lambda functions are identified as sync."""
        sync_lambda = lambda x: x * 2
        assert _is_async_function(sync_lambda) is False

    def test_identifies_method_as_sync(self):
        """Test that regular methods are identified as sync."""
        class MyClass:
            def method(self):
                return "result"

        obj = MyClass()
        assert _is_async_function(obj.method) is False

    def test_identifies_async_method(self):
        """Test that async methods are identified correctly."""
        class MyClass:
            async def async_method(self):
                return "result"

        obj = MyClass()
        assert _is_async_function(obj.async_method) is True

    def test_identifies_coroutine_function(self):
        """Test that coroutine functions are identified as async."""
        async def coro():
            await asyncio.sleep(0)
            return "done"

        assert _is_async_function(coro) is True


# =============================================================================
# Tests for _set_args_attributes()
# =============================================================================


class TestSetArgsAttributes:
    """Tests for the _set_args_attributes helper."""

    def test_sets_positional_args(self):
        """Test that positional args are set as span attributes."""
        mock_span = MagicMock()
        args = ("hello", 42, True)
        kwargs = {}

        _set_args_attributes(mock_span, args, kwargs, max_length=2000)

        mock_span.set_attribute.assert_any_call("function.arg.0", "hello")
        mock_span.set_attribute.assert_any_call("function.arg.1", "42")
        mock_span.set_attribute.assert_any_call("function.arg.2", "True")

    def test_sets_keyword_args(self):
        """Test that keyword args are set as span attributes."""
        mock_span = MagicMock()
        args = ()
        kwargs = {"name": "Alice", "age": 30}

        _set_args_attributes(mock_span, args, kwargs, max_length=2000)

        mock_span.set_attribute.assert_any_call("function.kwarg.name", "Alice")
        mock_span.set_attribute.assert_any_call("function.kwarg.age", "30")

    def test_sets_both_args_and_kwargs(self):
        """Test that both args and kwargs are set correctly."""
        mock_span = MagicMock()
        args = ("value1",)
        kwargs = {"key": "value2"}

        _set_args_attributes(mock_span, args, kwargs, max_length=2000)

        mock_span.set_attribute.assert_any_call("function.arg.0", "value1")
        mock_span.set_attribute.assert_any_call("function.kwarg.key", "value2")

    def test_empty_args_and_kwargs(self):
        """Test that empty args and kwargs result in no attribute calls."""
        mock_span = MagicMock()
        args = ()
        kwargs = {}

        _set_args_attributes(mock_span, args, kwargs, max_length=2000)

        mock_span.set_attribute.assert_not_called()

    def test_respects_max_length(self):
        """Test that max_length is passed to _safe_serialize."""
        mock_span = MagicMock()
        long_arg = "x" * 100
        args = (long_arg,)
        kwargs = {}

        _set_args_attributes(mock_span, args, kwargs, max_length=50)

        # The call should have been made with truncated value
        call_args = mock_span.set_attribute.call_args_list[0]
        attr_value = call_args[0][1]
        assert attr_value.endswith("...[truncated]")

    def test_serializes_complex_args(self):
        """Test that complex args are properly serialized."""
        mock_span = MagicMock()
        args = ([1, 2, 3],)
        kwargs = {"data": {"nested": True}}

        _set_args_attributes(mock_span, args, kwargs, max_length=2000)

        mock_span.set_attribute.assert_any_call("function.arg.0", "[1, 2, 3]")
        mock_span.set_attribute.assert_any_call(
            "function.kwarg.data", '{"nested": true}'
        )


# =============================================================================
# Tests for @traced decorator
# =============================================================================


class TestTracedDecorator:
    """Tests for the @traced decorator."""

    @pytest.fixture
    def mock_tracer(self):
        """Create a mock tracer with span context manager."""
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)

        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        return mock_tracer, mock_span

    def test_basic_sync_function_tracing(self, mock_tracer):
        """Test basic sync function tracing creates a span."""
        tracer, span = mock_tracer

        with patch("openbox.tracing._get_tracer", return_value=tracer):
            @traced
            def my_func(x):
                return x * 2

            result = my_func(5)

        assert result == 10
        tracer.start_as_current_span.assert_called_once_with("my_func")
        span.set_attribute.assert_any_call("code.function", "my_func")

    async def test_basic_async_function_tracing(self, mock_tracer):
        """Test basic async function tracing creates a span."""
        tracer, span = mock_tracer

        with patch("openbox.tracing._get_tracer", return_value=tracer):
            @traced
            async def my_async_func(x):
                return x * 2

            result = await my_async_func(5)

        assert result == 10
        tracer.start_as_current_span.assert_called_once_with("my_async_func")
        span.set_attribute.assert_any_call("code.function", "my_async_func")

    def test_traced_with_custom_name(self, mock_tracer):
        """Test @traced with custom name parameter."""
        tracer, span = mock_tracer

        with patch("openbox.tracing._get_tracer", return_value=tracer):
            @traced(name="custom-span-name")
            def my_func():
                return "result"

            my_func()

        tracer.start_as_current_span.assert_called_once_with("custom-span-name")
        # code.function should still be the actual function name
        span.set_attribute.assert_any_call("code.function", "my_func")

    def test_traced_capture_args_true(self, mock_tracer):
        """Test capture_args=True captures function arguments."""
        tracer, span = mock_tracer

        with patch("openbox.tracing._get_tracer", return_value=tracer):
            @traced(capture_args=True)
            def my_func(a, b, keyword=None):
                return a + b

            my_func(1, 2, keyword="test")

        span.set_attribute.assert_any_call("function.arg.0", "1")
        span.set_attribute.assert_any_call("function.arg.1", "2")
        span.set_attribute.assert_any_call("function.kwarg.keyword", "test")

    def test_traced_capture_args_false(self, mock_tracer):
        """Test capture_args=False does not capture arguments."""
        tracer, span = mock_tracer

        with patch("openbox.tracing._get_tracer", return_value=tracer):
            @traced(capture_args=False)
            def my_func(a, b):
                return a + b

            my_func(1, 2)

        # Check that no function.arg.* attributes were set
        for call_args in span.set_attribute.call_args_list:
            attr_name = call_args[0][0]
            assert not attr_name.startswith("function.arg.")
            assert not attr_name.startswith("function.kwarg.")

    def test_traced_capture_result_true(self, mock_tracer):
        """Test capture_result=True captures return value."""
        tracer, span = mock_tracer

        with patch("openbox.tracing._get_tracer", return_value=tracer):
            @traced(capture_result=True)
            def my_func():
                return {"status": "success", "data": [1, 2, 3]}

            my_func()

        span.set_attribute.assert_any_call(
            "function.result", '{"status": "success", "data": [1, 2, 3]}'
        )

    def test_traced_capture_result_false(self, mock_tracer):
        """Test capture_result=False does not capture return value."""
        tracer, span = mock_tracer

        with patch("openbox.tracing._get_tracer", return_value=tracer):
            @traced(capture_result=False)
            def my_func():
                return "secret result"

            my_func()

        # Check that function.result was not set
        for call_args in span.set_attribute.call_args_list:
            attr_name = call_args[0][0]
            assert attr_name != "function.result"

    def test_traced_capture_exception_true(self, mock_tracer):
        """Test capture_exception=True records error attributes."""
        tracer, span = mock_tracer

        with patch("openbox.tracing._get_tracer", return_value=tracer):
            @traced(capture_exception=True)
            def failing_func():
                raise ValueError("Something went wrong")

            with pytest.raises(ValueError, match="Something went wrong"):
                failing_func()

        span.set_attribute.assert_any_call("error", True)
        span.set_attribute.assert_any_call("error.type", "ValueError")
        span.set_attribute.assert_any_call("error.message", "Something went wrong")

    def test_traced_capture_exception_false(self, mock_tracer):
        """Test capture_exception=False does not record error attributes."""
        tracer, span = mock_tracer

        with patch("openbox.tracing._get_tracer", return_value=tracer):
            @traced(capture_exception=False)
            def failing_func():
                raise RuntimeError("Error occurred")

            with pytest.raises(RuntimeError, match="Error occurred"):
                failing_func()

        # Check that error attributes were not set
        for call_args in span.set_attribute.call_args_list:
            attr_name = call_args[0][0]
            assert attr_name not in ("error", "error.type", "error.message")

    def test_exception_is_reraised(self, mock_tracer):
        """Test that exceptions are re-raised after being captured."""
        tracer, span = mock_tracer

        with patch("openbox.tracing._get_tracer", return_value=tracer):
            @traced
            def failing_func():
                raise KeyError("missing key")

            with pytest.raises(KeyError, match="missing key"):
                failing_func()

    async def test_async_exception_handling(self, mock_tracer):
        """Test exception handling in async functions."""
        tracer, span = mock_tracer

        with patch("openbox.tracing._get_tracer", return_value=tracer):
            @traced(capture_exception=True)
            async def async_failing_func():
                raise TypeError("async type error")

            with pytest.raises(TypeError, match="async type error"):
                await async_failing_func()

        span.set_attribute.assert_any_call("error", True)
        span.set_attribute.assert_any_call("error.type", "TypeError")
        span.set_attribute.assert_any_call("error.message", "async type error")

    def test_traced_bare_decorator_syntax(self, mock_tracer):
        """Test @traced syntax (without parentheses) works."""
        tracer, span = mock_tracer

        with patch("openbox.tracing._get_tracer", return_value=tracer):
            @traced
            def bare_decorated():
                return "bare"

            result = bare_decorated()

        assert result == "bare"
        tracer.start_as_current_span.assert_called_once_with("bare_decorated")

    def test_traced_called_decorator_syntax(self, mock_tracer):
        """Test @traced() syntax (with parentheses) works."""
        tracer, span = mock_tracer

        with patch("openbox.tracing._get_tracer", return_value=tracer):
            @traced()
            def called_decorated():
                return "called"

            result = called_decorated()

        assert result == "called"
        tracer.start_as_current_span.assert_called_once_with("called_decorated")

    def test_traced_preserves_function_metadata(self, mock_tracer):
        """Test that @traced preserves function name and docstring."""
        tracer, span = mock_tracer

        with patch("openbox.tracing._get_tracer", return_value=tracer):
            @traced
            def documented_func(x):
                """This is a documented function."""
                return x

            assert documented_func.__name__ == "documented_func"
            assert documented_func.__doc__ == "This is a documented function."

    def test_traced_sets_code_namespace(self, mock_tracer):
        """Test that traced sets code.namespace attribute."""
        tracer, span = mock_tracer

        with patch("openbox.tracing._get_tracer", return_value=tracer):
            @traced
            def namespaced_func():
                return "result"

            namespaced_func()

        # Find the code.namespace call
        namespace_calls = [
            c for c in span.set_attribute.call_args_list
            if c[0][0] == "code.namespace"
        ]
        assert len(namespace_calls) == 1
        # The namespace should contain the module name (may include package path)
        assert "test_tracing" in namespace_calls[0][0][1]

    async def test_async_traced_with_all_options(self, mock_tracer):
        """Test async function with all traced options enabled."""
        tracer, span = mock_tracer

        with patch("openbox.tracing._get_tracer", return_value=tracer):
            @traced(
                name="async-operation",
                capture_args=True,
                capture_result=True,
                capture_exception=True,
            )
            async def async_operation(data, multiplier=2):
                return data * multiplier

            result = await async_operation("test", multiplier=3)

        assert result == "testtesttest"
        tracer.start_as_current_span.assert_called_once_with("async-operation")
        span.set_attribute.assert_any_call("function.arg.0", "test")
        span.set_attribute.assert_any_call("function.kwarg.multiplier", "3")
        span.set_attribute.assert_any_call("function.result", "testtesttest")

    def test_traced_with_none_return_value(self, mock_tracer):
        """Test that None return values are captured as 'null'."""
        tracer, span = mock_tracer

        with patch("openbox.tracing._get_tracer", return_value=tracer):
            @traced(capture_result=True)
            def returns_none():
                return None

            result = returns_none()

        assert result is None
        span.set_attribute.assert_any_call("function.result", "null")

    def test_traced_with_max_arg_length(self, mock_tracer):
        """Test that max_arg_length parameter is respected."""
        tracer, span = mock_tracer

        with patch("openbox.tracing._get_tracer", return_value=tracer):
            @traced(capture_args=True, capture_result=True, max_arg_length=50)
            def long_args_func(data):
                return data

            long_data = "x" * 100
            long_args_func(long_data)

        # Check that arg was truncated
        arg_calls = [
            c for c in span.set_attribute.call_args_list
            if c[0][0] == "function.arg.0"
        ]
        assert len(arg_calls) == 1
        assert arg_calls[0][0][1].endswith("...[truncated]")


# =============================================================================
# Tests for create_span()
# =============================================================================


class TestCreateSpan:
    """Tests for the create_span function."""

    def test_creates_span_with_name(self):
        """Test that create_span creates a span with the given name."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_span.return_value = mock_span

        with patch("openbox.tracing._get_tracer", return_value=mock_tracer):
            span = create_span("test-span-name")

        mock_tracer.start_span.assert_called_once_with("test-span-name")
        assert span == mock_span

    def test_creates_span_with_attributes(self):
        """Test that create_span sets initial attributes."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_span.return_value = mock_span

        with patch("openbox.tracing._get_tracer", return_value=mock_tracer):
            span = create_span(
                "operation",
                attributes={"user_id": "12345", "action": "create"}
            )

        mock_span.set_attribute.assert_any_call("user_id", "12345")
        mock_span.set_attribute.assert_any_call("action", "create")

    def test_creates_span_without_attributes(self):
        """Test that create_span works without attributes."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_span.return_value = mock_span

        with patch("openbox.tracing._get_tracer", return_value=mock_tracer):
            span = create_span("simple-span")

        mock_tracer.start_span.assert_called_once_with("simple-span")
        mock_span.set_attribute.assert_not_called()

    def test_span_attributes_are_serialized(self):
        """Test that attribute values are serialized with _safe_serialize."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_span.return_value = mock_span

        with patch("openbox.tracing._get_tracer", return_value=mock_tracer):
            span = create_span(
                "complex-span",
                attributes={"data": {"nested": [1, 2, 3]}, "count": 42}
            )

        # Check that dict was JSON serialized
        data_calls = [
            c for c in mock_span.set_attribute.call_args_list
            if c[0][0] == "data"
        ]
        assert len(data_calls) == 1
        assert data_calls[0][0][1] == '{"nested": [1, 2, 3]}'

        # Check that int was string serialized
        count_calls = [
            c for c in mock_span.set_attribute.call_args_list
            if c[0][0] == "count"
        ]
        assert len(count_calls) == 1
        assert count_calls[0][0][1] == "42"

    def test_span_can_be_used_as_context_manager(self):
        """Test that the returned span can be used as a context manager."""
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)

        mock_tracer = MagicMock()
        mock_tracer.start_span.return_value = mock_span

        with patch("openbox.tracing._get_tracer", return_value=mock_tracer):
            with create_span("context-span") as span:
                span.set_attribute("inside", "context")

        mock_span.__enter__.assert_called_once()
        mock_span.__exit__.assert_called_once()
        mock_span.set_attribute.assert_any_call("inside", "context")

    def test_create_span_with_none_value_in_attributes(self):
        """Test that None values in attributes are handled correctly."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_span.return_value = mock_span

        with patch("openbox.tracing._get_tracer", return_value=mock_tracer):
            span = create_span("span-with-none", attributes={"nullable": None})

        mock_span.set_attribute.assert_any_call("nullable", "null")


# =============================================================================
# Tests for _get_tracer()
# =============================================================================


class TestGetTracer:
    """Tests for the _get_tracer function."""

    def test_get_tracer_returns_tracer(self):
        """Test that _get_tracer returns an OpenTelemetry tracer."""
        # Reset the global tracer
        import openbox.tracing
        openbox.tracing._tracer = None

        with patch("openbox.tracing.trace.get_tracer") as mock_get_tracer:
            mock_tracer = MagicMock()
            mock_get_tracer.return_value = mock_tracer

            result = _get_tracer()

            mock_get_tracer.assert_called_once_with("openbox.traced")
            assert result == mock_tracer

    def test_get_tracer_caches_tracer(self):
        """Test that _get_tracer caches the tracer instance."""
        import openbox.tracing
        openbox.tracing._tracer = None

        with patch("openbox.tracing.trace.get_tracer") as mock_get_tracer:
            mock_tracer = MagicMock()
            mock_get_tracer.return_value = mock_tracer

            # Call twice
            result1 = _get_tracer()
            result2 = _get_tracer()

            # Should only call get_tracer once
            mock_get_tracer.assert_called_once()
            assert result1 is result2


# =============================================================================
# Integration-style tests
# =============================================================================


class TestTracedIntegration:
    """Integration-style tests for traced decorator behavior."""

    @pytest.fixture(autouse=True)
    def reset_tracer(self):
        """Reset the global tracer before each test."""
        import openbox.tracing
        openbox.tracing._tracer = None
        yield
        openbox.tracing._tracer = None

    def test_traced_function_returns_correct_value(self):
        """Test that traced functions return correct values."""
        @traced
        def add(a, b):
            return a + b

        assert add(2, 3) == 5
        assert add(-1, 1) == 0
        assert add(100, 200) == 300

    async def test_traced_async_function_returns_correct_value(self):
        """Test that traced async functions return correct values."""
        @traced
        async def async_multiply(a, b):
            await asyncio.sleep(0)  # Simulate async operation
            return a * b

        assert await async_multiply(3, 4) == 12
        assert await async_multiply(0, 100) == 0

    def test_traced_function_with_side_effects(self):
        """Test that traced functions can have side effects."""
        results = []

        @traced
        def append_to_results(value):
            results.append(value)
            return len(results)

        assert append_to_results("a") == 1
        assert append_to_results("b") == 2
        assert results == ["a", "b"]

    def test_multiple_traced_functions(self):
        """Test multiple traced functions can work together."""
        @traced(name="step-1")
        def step_one(x):
            return x + 1

        @traced(name="step-2")
        def step_two(x):
            return x * 2

        @traced(name="pipeline")
        def pipeline(x):
            return step_two(step_one(x))

        assert pipeline(5) == 12  # (5 + 1) * 2

    async def test_nested_async_traced_functions(self):
        """Test nested async traced functions."""
        @traced
        async def inner():
            return "inner"

        @traced
        async def outer():
            result = await inner()
            return f"outer({result})"

        assert await outer() == "outer(inner)"

    def test_traced_generator_function(self):
        """Test that traced works with functions that return generators."""
        @traced
        def get_range(n):
            return range(n)

        result = get_range(5)
        assert list(result) == [0, 1, 2, 3, 4]

    def test_traced_class_method(self):
        """Test that traced works with class methods."""
        class Calculator:
            @traced(name="calculator-add")
            def add(self, a, b):
                return a + b

            @traced(name="calculator-multiply")
            def multiply(self, a, b):
                return a * b

        calc = Calculator()
        assert calc.add(2, 3) == 5
        assert calc.multiply(4, 5) == 20

    async def test_traced_async_class_method(self):
        """Test that traced works with async class methods."""
        class AsyncProcessor:
            @traced
            async def process(self, data):
                await asyncio.sleep(0)
                return data.upper()

        processor = AsyncProcessor()
        assert await processor.process("hello") == "HELLO"

    def test_traced_with_defaults_disabled(self):
        """Test traced with all capture options disabled."""
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)

        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        with patch("openbox.tracing._get_tracer", return_value=mock_tracer):
            @traced(capture_args=False, capture_result=False, capture_exception=False)
            def minimal_func(a, b):
                return a + b

            result = minimal_func(1, 2)

        assert result == 3

        # Should only have code.function and code.namespace attributes
        attr_names = [c[0][0] for c in mock_span.set_attribute.call_args_list]
        assert "code.function" in attr_names
        assert "code.namespace" in attr_names
        assert "function.arg.0" not in attr_names
        assert "function.result" not in attr_names


# =============================================================================
# Governance test helpers
# =============================================================================


@pytest.fixture
def cleanup_governance():
    """Reset hook_governance module state after each test."""
    yield
    hook_gov._api_url = ""
    hook_gov._api_key = ""
    hook_gov._span_processor = None


def _setup_governance(on_api_error: str = "fail_open") -> MagicMock:
    """Set up hook_governance with a mock span processor."""
    processor = MagicMock()
    processor.get_activity_context_by_trace.return_value = {
        "workflow_id": "wf-traced-1",
        "activity_id": "act-traced-1",
    }
    buffer = WorkflowSpanBuffer(
        workflow_id="wf-traced-1", run_id="run-1",
        workflow_type="TracedWorkflow", task_queue="traced-queue",
    )
    processor.get_buffer.return_value = buffer

    hook_gov.configure(
        "http://localhost:9090", "test-key", processor,
        api_timeout=5.0, on_api_error=on_api_error,
    )
    return processor


@contextmanager
def _mock_httpx_client(verdict="allow", reason=None, side_effect=None):
    """Mock httpx.Client for governance API calls."""
    response = MagicMock()
    response.status_code = 200
    response_data = {"verdict": verdict}
    if reason:
        response_data["reason"] = reason
    response.json.return_value = response_data

    mock_instance = MagicMock()
    if side_effect:
        mock_instance.__enter__ = MagicMock(side_effect=side_effect)
    else:
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.post.return_value = response
    mock_instance.__exit__ = MagicMock(return_value=False)

    with patch("openbox.hook_governance.httpx.Client", return_value=mock_instance):
        yield mock_instance


@contextmanager
def _mock_httpx_async_client(verdict="allow", reason=None, side_effect=None):
    """Mock httpx.AsyncClient for async governance API calls."""
    response = MagicMock()
    response.status_code = 200
    response_data = {"verdict": verdict}
    if reason:
        response_data["reason"] = reason
    response.json.return_value = response_data

    mock_instance = MagicMock()
    if side_effect:
        mock_instance.__aenter__ = AsyncMock(side_effect=side_effect)
    else:
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.post = AsyncMock(return_value=response)
    mock_instance.__aexit__ = AsyncMock(return_value=None)

    with patch("openbox.hook_governance.httpx.AsyncClient", return_value=mock_instance):
        yield mock_instance


# =============================================================================
# Tests for @traced governance: started stage
# =============================================================================


class TestTracedGovernanceStarted:
    """Tests for governance 'started' stage on @traced decorator."""

    @pytest.fixture(autouse=True)
    def setup(self, cleanup_governance):
        """Set up governance for each test."""
        pass

    def test_governed_sends_started_before_execution(self):
        """Verify started hook_trigger is sent with function name, module, stage."""
        _setup_governance()
        executed = []

        @traced
        def my_func(x):
            executed.append(True)
            return x * 2

        with _mock_httpx_client(verdict="allow") as mock_client:
            result = my_func(5)

        assert result == 10
        assert executed == [True]

        # Should have 2 governance calls (started + completed)
        calls = mock_client.post.call_args_list
        assert len(calls) == 2

        # Check started payload
        started_payload = calls[0].kwargs.get("json") or calls[0][1].get("json")
        trigger = started_payload["spans"][0]
        assert trigger["hook_type"] == "function_call"
        assert trigger["function"] == "my_func"
        assert trigger["stage"] == "started"

    def test_governed_started_block_prevents_execution(self):
        """BLOCK verdict at started → function never runs."""
        _setup_governance()
        executed = []

        @traced
        def blocked_func():
            executed.append(True)
            return "should not return"

        with _mock_httpx_client(verdict="block", reason="Policy violation"):
            with pytest.raises(GovernanceBlockedError) as exc_info:
                blocked_func()

        assert executed == []
        assert "block" in exc_info.value.verdict

    def test_governed_started_halt_prevents_execution(self):
        """HALT verdict at started → function never runs."""
        _setup_governance()
        executed = []

        @traced
        def halted_func():
            executed.append(True)

        with _mock_httpx_client(verdict="halt"):
            with pytest.raises(GovernanceBlockedError) as exc_info:
                halted_func()

        assert executed == []
        assert "halt" in exc_info.value.verdict

    def test_governed_started_includes_args(self):
        """When capture_args=True, args appear in hook_trigger."""
        _setup_governance()

        @traced(capture_args=True)
        def func_with_args(a, b):
            return a + b

        with _mock_httpx_client(verdict="allow") as mock_client:
            func_with_args(1, 2)

        calls = mock_client.post.call_args_list
        started_payload = calls[0].kwargs.get("json") or calls[0][1].get("json")
        trigger = started_payload["spans"][0]
        assert "args" in trigger


# =============================================================================
# Tests for @traced governance: completed stage
# =============================================================================


class TestTracedGovernanceCompleted:
    """Tests for governance 'completed' stage on @traced decorator."""

    @pytest.fixture(autouse=True)
    def setup(self, cleanup_governance):
        pass

    def test_governed_sends_completed_after_execution(self):
        """Verify completed hook_trigger sent after execution."""
        _setup_governance()

        @traced
        def my_func():
            return "done"

        with _mock_httpx_client(verdict="allow") as mock_client:
            my_func()

        calls = mock_client.post.call_args_list
        assert len(calls) == 2

        completed_payload = calls[1].kwargs.get("json") or calls[1][1].get("json")
        trigger = completed_payload["spans"][0]
        assert trigger["hook_type"] == "function_call"
        assert trigger["stage"] == "completed"

    def test_governed_completed_includes_result(self):
        """When capture_result=True, result appears in completed hook_trigger."""
        _setup_governance()

        @traced(capture_result=True)
        def func_with_result():
            return {"status": "ok"}

        with _mock_httpx_client(verdict="allow") as mock_client:
            func_with_result()

        calls = mock_client.post.call_args_list
        completed_payload = calls[1].kwargs.get("json") or calls[1][1].get("json")
        trigger = completed_payload["spans"][0]
        assert "result" in trigger

    def test_governed_completed_includes_error_on_exception(self):
        """Error info in completed hook_trigger when function raises."""
        _setup_governance()

        @traced
        def failing_func():
            raise ValueError("test error")

        with _mock_httpx_client(verdict="allow") as mock_client:
            with pytest.raises(ValueError, match="test error"):
                failing_func()

        calls = mock_client.post.call_args_list
        assert len(calls) == 2

        error_payload = calls[1].kwargs.get("json") or calls[1][1].get("json")
        trigger = error_payload["spans"][0]
        assert trigger["stage"] == "completed"
        assert trigger["error"] == "test error"

    def test_governed_started_and_completed_both_sent(self):
        """2 governance calls per invocation (started + completed)."""
        _setup_governance()

        @traced
        def simple_func():
            return 42

        with _mock_httpx_client(verdict="allow") as mock_client:
            simple_func()

        calls = mock_client.post.call_args_list
        assert len(calls) == 2

        stages = []
        for c in calls:
            payload = c.kwargs.get("json") or c[1].get("json")
            stages.append(payload["spans"][0]["stage"])
        assert stages == ["started", "completed"]


# =============================================================================
# Tests for @traced governance: async path
# =============================================================================


class TestTracedGovernanceAsync:
    """Tests for governance on async @traced functions."""

    @pytest.fixture(autouse=True)
    def setup(self, cleanup_governance):
        pass

    async def test_async_governed_sends_started_and_completed(self):
        """Async path sends both started and completed stages."""
        _setup_governance()

        @traced
        async def async_func():
            return "async result"

        with _mock_httpx_async_client(verdict="allow") as mock_client:
            result = await async_func()

        assert result == "async result"
        calls = mock_client.post.call_args_list
        assert len(calls) == 2

        stages = []
        for c in calls:
            payload = c[0][1] if len(c[0]) > 1 else c.kwargs.get("json")
            stages.append(payload["spans"][0]["stage"])
        assert stages == ["started", "completed"]

    async def test_async_governed_block_prevents_execution(self):
        """Async BLOCK at started prevents execution."""
        _setup_governance()
        executed = []

        @traced
        async def blocked_async():
            executed.append(True)
            return "should not return"

        with _mock_httpx_async_client(verdict="block"):
            with pytest.raises(GovernanceBlockedError):
                await blocked_async()

        assert executed == []


# =============================================================================
# Tests for @traced governance: not configured
# =============================================================================


class TestTracedGovernanceNotConfigured:
    """Tests for @traced when governance is not configured."""

    @pytest.fixture(autouse=True)
    def setup(self, cleanup_governance):
        pass

    def test_no_governance_calls_when_not_configured(self):
        """No governance calls when hook_governance not configured."""
        # Don't call _setup_governance() — leave unconfigured

        @traced
        def my_func():
            return "no governance"

        with _mock_httpx_client(verdict="allow") as mock_client:
            result = my_func()

        assert result == "no governance"
        mock_client.post.assert_not_called()

    def test_existing_traced_behavior_unchanged(self):
        """Existing @traced behavior works identically without governance."""
        @traced
        def add(a, b):
            return a + b

        assert add(2, 3) == 5
        assert add(-1, 1) == 0


# =============================================================================
# Tests for @traced governance: fail policies
# =============================================================================


class TestTracedGovernanceFailPolicy:
    """Tests for fail_open/fail_closed policies on @traced governance."""

    @pytest.fixture(autouse=True)
    def setup(self, cleanup_governance):
        pass

    def test_governed_fail_open_allows_on_api_error(self):
        """fail_open: function runs despite API error."""
        _setup_governance(on_api_error="fail_open")

        @traced
        def my_func():
            return "success"

        with _mock_httpx_client(side_effect=ConnectionError("API down")):
            result = my_func()

        assert result == "success"

    def test_governed_fail_closed_blocks_on_api_error(self):
        """fail_closed: GovernanceBlockedError on API error."""
        _setup_governance(on_api_error="fail_closed")

        @traced
        def my_func():
            return "should not return"

        with _mock_httpx_client(side_effect=ConnectionError("API down")):
            with pytest.raises(GovernanceBlockedError):
                my_func()
