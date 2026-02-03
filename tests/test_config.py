# tests/test_config.py
"""
Comprehensive pytest tests for the OpenBox SDK config module.

Tests cover:
- GovernanceConfig dataclass (defaults, custom values)
- _validate_api_key_format() function
- _GlobalConfig class
- get_global_config() singleton
- initialize() function
- Exception classes hierarchy
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from urllib.error import HTTPError, URLError

from openbox.config import (
    GovernanceConfig,
    _validate_api_key_format,
    _validate_url_security,
    _GlobalConfig,
    get_global_config,
    initialize,
    OpenBoxConfigError,
    OpenBoxAuthError,
    OpenBoxNetworkError,
    OpenBoxInsecureURLError,
    API_KEY_PATTERN,
)


# ═══════════════════════════════════════════════════════════════════════════════
# GovernanceConfig Dataclass Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGovernanceConfigDefaults:
    """Test GovernanceConfig default values."""

    def test_default_on_api_error(self):
        """Test default on_api_error is 'fail_open'."""
        config = GovernanceConfig()
        assert config.on_api_error == "fail_open"

    def test_default_api_timeout(self):
        """Test default api_timeout is 30.0 seconds."""
        config = GovernanceConfig()
        assert config.api_timeout == 30.0

    def test_default_send_start_event(self):
        """Test default send_start_event is True."""
        config = GovernanceConfig()
        assert config.send_start_event is True

    def test_default_send_activity_start_event(self):
        """Test default send_activity_start_event is True."""
        config = GovernanceConfig()
        assert config.send_activity_start_event is True

    def test_default_skip_workflow_types(self):
        """Test default skip_workflow_types is empty set."""
        config = GovernanceConfig()
        assert config.skip_workflow_types == set()
        assert isinstance(config.skip_workflow_types, set)

    def test_default_skip_signals(self):
        """Test default skip_signals is empty set."""
        config = GovernanceConfig()
        assert config.skip_signals == set()
        assert isinstance(config.skip_signals, set)

    def test_default_enforce_task_queues(self):
        """Test default enforce_task_queues is None (all queues)."""
        config = GovernanceConfig()
        assert config.enforce_task_queues is None

    def test_default_max_body_size(self):
        """Test default max_body_size is None (no limit)."""
        config = GovernanceConfig()
        assert config.max_body_size is None

    def test_default_skip_activity_types_includes_governance_event(self):
        """Test default skip_activity_types includes 'send_governance_event'."""
        config = GovernanceConfig()
        assert "send_governance_event" in config.skip_activity_types
        assert isinstance(config.skip_activity_types, set)

    def test_default_hitl_enabled(self):
        """Test default hitl_enabled is True."""
        config = GovernanceConfig()
        assert config.hitl_enabled is True

    def test_default_skip_hitl_activity_types_includes_governance_event(self):
        """Test default skip_hitl_activity_types includes 'send_governance_event'."""
        config = GovernanceConfig()
        assert "send_governance_event" in config.skip_hitl_activity_types
        assert isinstance(config.skip_hitl_activity_types, set)


class TestGovernanceConfigCustomValues:
    """Test GovernanceConfig with custom values."""

    def test_custom_on_api_error_fail_closed(self):
        """Test setting on_api_error to 'fail_closed'."""
        config = GovernanceConfig(on_api_error="fail_closed")
        assert config.on_api_error == "fail_closed"

    def test_custom_api_timeout(self):
        """Test setting custom api_timeout."""
        config = GovernanceConfig(api_timeout=60.0)
        assert config.api_timeout == 60.0

    def test_custom_api_timeout_zero(self):
        """Test setting api_timeout to zero."""
        config = GovernanceConfig(api_timeout=0.0)
        assert config.api_timeout == 0.0

    def test_custom_send_start_event_false(self):
        """Test disabling send_start_event."""
        config = GovernanceConfig(send_start_event=False)
        assert config.send_start_event is False

    def test_custom_send_activity_start_event_false(self):
        """Test disabling send_activity_start_event."""
        config = GovernanceConfig(send_activity_start_event=False)
        assert config.send_activity_start_event is False

    def test_custom_skip_workflow_types(self):
        """Test setting custom skip_workflow_types."""
        skip_types = {"WorkflowA", "WorkflowB"}
        config = GovernanceConfig(skip_workflow_types=skip_types)
        assert config.skip_workflow_types == skip_types

    def test_custom_skip_signals(self):
        """Test setting custom skip_signals."""
        skip_signals = {"signal_a", "signal_b"}
        config = GovernanceConfig(skip_signals=skip_signals)
        assert config.skip_signals == skip_signals

    def test_custom_enforce_task_queues(self):
        """Test setting enforce_task_queues to specific queues."""
        queues = {"queue-1", "queue-2"}
        config = GovernanceConfig(enforce_task_queues=queues)
        assert config.enforce_task_queues == queues

    def test_custom_max_body_size(self):
        """Test setting max_body_size."""
        config = GovernanceConfig(max_body_size=1024)
        assert config.max_body_size == 1024

    def test_custom_skip_activity_types(self):
        """Test setting custom skip_activity_types replaces default."""
        custom_skip = {"my_activity", "another_activity"}
        config = GovernanceConfig(skip_activity_types=custom_skip)
        assert config.skip_activity_types == custom_skip
        # Default "send_governance_event" is NOT included when custom is provided
        assert "send_governance_event" not in config.skip_activity_types

    def test_custom_skip_activity_types_with_governance_event(self):
        """Test custom skip_activity_types can include governance event."""
        custom_skip = {"send_governance_event", "my_activity"}
        config = GovernanceConfig(skip_activity_types=custom_skip)
        assert config.skip_activity_types == custom_skip

    def test_custom_hitl_enabled_false(self):
        """Test disabling hitl_enabled."""
        config = GovernanceConfig(hitl_enabled=False)
        assert config.hitl_enabled is False

    def test_custom_skip_hitl_activity_types(self):
        """Test setting custom skip_hitl_activity_types."""
        custom_skip = {"poll_approval", "my_hitl_activity"}
        config = GovernanceConfig(skip_hitl_activity_types=custom_skip)
        assert config.skip_hitl_activity_types == custom_skip

    def test_all_custom_values(self):
        """Test creating config with all custom values."""
        config = GovernanceConfig(
            skip_workflow_types={"WF1", "WF2"},
            skip_signals={"sig1"},
            enforce_task_queues={"queue1"},
            on_api_error="fail_closed",
            api_timeout=45.0,
            max_body_size=2048,
            send_start_event=False,
            send_activity_start_event=False,
            skip_activity_types={"activity1"},
            hitl_enabled=False,
            skip_hitl_activity_types={"hitl1"},
        )
        assert config.skip_workflow_types == {"WF1", "WF2"}
        assert config.skip_signals == {"sig1"}
        assert config.enforce_task_queues == {"queue1"}
        assert config.on_api_error == "fail_closed"
        assert config.api_timeout == 45.0
        assert config.max_body_size == 2048
        assert config.send_start_event is False
        assert config.send_activity_start_event is False
        assert config.skip_activity_types == {"activity1"}
        assert config.hitl_enabled is False
        assert config.skip_hitl_activity_types == {"hitl1"}


class TestGovernanceConfigMutability:
    """Test GovernanceConfig mutable default factory behavior."""

    def test_skip_workflow_types_not_shared_between_instances(self):
        """Test that default skip_workflow_types is not shared."""
        config1 = GovernanceConfig()
        config2 = GovernanceConfig()
        config1.skip_workflow_types.add("WorkflowX")
        assert "WorkflowX" not in config2.skip_workflow_types

    def test_skip_activity_types_not_shared_between_instances(self):
        """Test that default skip_activity_types is not shared."""
        config1 = GovernanceConfig()
        config2 = GovernanceConfig()
        config1.skip_activity_types.add("ActivityX")
        assert "ActivityX" not in config2.skip_activity_types

    def test_skip_hitl_activity_types_not_shared_between_instances(self):
        """Test that default skip_hitl_activity_types is not shared."""
        config1 = GovernanceConfig()
        config2 = GovernanceConfig()
        config1.skip_hitl_activity_types.add("HitlX")
        assert "HitlX" not in config2.skip_hitl_activity_types


# ═══════════════════════════════════════════════════════════════════════════════
# _validate_url_security() Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateUrlSecurity:
    """Test _validate_url_security() function - HTTPS enforcement for non-localhost."""

    # Valid URLs (should not raise)
    def test_https_url_passes(self):
        """Test HTTPS URL passes validation."""
        _validate_url_security("https://api.openbox.ai")

    def test_https_url_with_port_passes(self):
        """Test HTTPS URL with port passes validation."""
        _validate_url_security("https://api.openbox.ai:8443")

    def test_http_localhost_passes(self):
        """Test HTTP localhost passes validation."""
        _validate_url_security("http://localhost:8086")

    def test_http_localhost_no_port_passes(self):
        """Test HTTP localhost without port passes validation."""
        _validate_url_security("http://localhost")

    def test_http_127_0_0_1_passes(self):
        """Test HTTP 127.0.0.1 passes validation."""
        _validate_url_security("http://127.0.0.1:8086")

    def test_http_127_0_0_1_no_port_passes(self):
        """Test HTTP 127.0.0.1 without port passes validation."""
        _validate_url_security("http://127.0.0.1")

    def test_http_ipv6_localhost_passes(self):
        """Test HTTP IPv6 localhost (::1) passes validation."""
        _validate_url_security("http://[::1]:8086")

    # Invalid URLs (should raise OpenBoxInsecureURLError)
    def test_http_external_url_fails(self):
        """Test HTTP external URL raises OpenBoxInsecureURLError."""
        with pytest.raises(OpenBoxInsecureURLError) as exc_info:
            _validate_url_security("http://api.openbox.ai")
        assert "Insecure HTTP URL detected" in str(exc_info.value)
        assert "HTTPS" in str(exc_info.value)

    def test_http_external_ip_fails(self):
        """Test HTTP external IP raises OpenBoxInsecureURLError."""
        with pytest.raises(OpenBoxInsecureURLError) as exc_info:
            _validate_url_security("http://192.168.1.100:8086")
        assert "Insecure HTTP URL detected" in str(exc_info.value)

    def test_http_external_domain_with_port_fails(self):
        """Test HTTP external domain with port raises OpenBoxInsecureURLError."""
        with pytest.raises(OpenBoxInsecureURLError) as exc_info:
            _validate_url_security("http://api.example.com:8086")
        assert "Insecure HTTP URL detected" in str(exc_info.value)

    def test_http_subdomain_fails(self):
        """Test HTTP subdomain raises OpenBoxInsecureURLError."""
        with pytest.raises(OpenBoxInsecureURLError) as exc_info:
            _validate_url_security("http://staging.api.openbox.ai")
        assert "Insecure HTTP URL detected" in str(exc_info.value)


# ═══════════════════════════════════════════════════════════════════════════════
# _validate_api_key_format() Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateApiKeyFormat:
    """Test _validate_api_key_format() function."""

    # Valid formats
    def test_valid_live_key_simple(self):
        """Test valid live API key format."""
        assert _validate_api_key_format("obx_live_abc123") is True

    def test_valid_test_key_simple(self):
        """Test valid test API key format."""
        assert _validate_api_key_format("obx_test_xyz789") is True

    def test_valid_live_key_long(self):
        """Test valid live key with long suffix."""
        assert _validate_api_key_format("obx_live_abcdefghijklmnopqrstuvwxyz123456") is True

    def test_valid_test_key_long(self):
        """Test valid test key with long suffix."""
        assert _validate_api_key_format("obx_test_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456") is True

    def test_valid_key_with_underscores_in_suffix(self):
        """Test valid key with underscores in suffix."""
        assert _validate_api_key_format("obx_live_abc_def_ghi") is True

    def test_valid_key_mixed_case(self):
        """Test valid key with mixed case suffix."""
        assert _validate_api_key_format("obx_test_AbCdEf123") is True

    def test_valid_key_numbers_only_suffix(self):
        """Test valid key with numbers only suffix."""
        assert _validate_api_key_format("obx_live_123456789") is True

    def test_valid_key_single_char_suffix(self):
        """Test valid key with single character suffix."""
        assert _validate_api_key_format("obx_test_a") is True

    # Invalid formats
    def test_invalid_completely_wrong_format(self):
        """Test completely invalid format."""
        assert _validate_api_key_format("invalid") is False

    def test_invalid_missing_suffix(self):
        """Test key missing suffix after environment."""
        assert _validate_api_key_format("obx_live") is False

    def test_invalid_missing_suffix_with_trailing_underscore(self):
        """Test key with trailing underscore but no suffix."""
        assert _validate_api_key_format("obx_live_") is False

    def test_invalid_missing_environment(self):
        """Test key missing environment (live/test)."""
        assert _validate_api_key_format("obx_abc123") is False

    def test_invalid_wrong_prefix(self):
        """Test key with wrong prefix."""
        assert _validate_api_key_format("api_live_abc123") is False

    def test_invalid_no_prefix(self):
        """Test key with no prefix."""
        assert _validate_api_key_format("live_abc123") is False

    def test_invalid_wrong_environment(self):
        """Test key with invalid environment."""
        assert _validate_api_key_format("obx_prod_abc123") is False

    def test_invalid_environment_dev(self):
        """Test key with dev environment (invalid)."""
        assert _validate_api_key_format("obx_dev_abc123") is False

    def test_invalid_empty_string(self):
        """Test empty string."""
        assert _validate_api_key_format("") is False

    def test_invalid_whitespace_only(self):
        """Test whitespace only."""
        assert _validate_api_key_format("   ") is False

    def test_invalid_with_leading_whitespace(self):
        """Test key with leading whitespace."""
        assert _validate_api_key_format(" obx_live_abc123") is False

    def test_invalid_with_trailing_whitespace(self):
        """Test key with trailing whitespace."""
        assert _validate_api_key_format("obx_live_abc123 ") is False

    def test_invalid_with_special_chars_in_suffix(self):
        """Test key with special characters in suffix."""
        assert _validate_api_key_format("obx_live_abc!@#") is False

    def test_invalid_uppercase_prefix(self):
        """Test key with uppercase prefix."""
        assert _validate_api_key_format("OBX_live_abc123") is False

    def test_invalid_uppercase_environment(self):
        """Test key with uppercase environment."""
        assert _validate_api_key_format("obx_LIVE_abc123") is False

    def test_invalid_with_dash_separator(self):
        """Test key using dash instead of underscore."""
        assert _validate_api_key_format("obx-live-abc123") is False


class TestApiKeyPattern:
    """Test the API_KEY_PATTERN regex directly."""

    def test_pattern_matches_live(self):
        """Test pattern matches live keys."""
        assert API_KEY_PATTERN.match("obx_live_abc") is not None

    def test_pattern_matches_test(self):
        """Test pattern matches test keys."""
        assert API_KEY_PATTERN.match("obx_test_xyz") is not None

    def test_pattern_no_match_staging(self):
        """Test pattern does not match staging."""
        assert API_KEY_PATTERN.match("obx_staging_abc") is None


# ═══════════════════════════════════════════════════════════════════════════════
# _GlobalConfig Class Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGlobalConfigClass:
    """Test _GlobalConfig class."""

    def test_init_defaults(self):
        """Test _GlobalConfig initializes with empty defaults."""
        config = _GlobalConfig()
        assert config.api_url == ""
        assert config.api_key == ""
        assert config.governance_timeout == 30.0

    def test_configure_sets_values(self):
        """Test configure() sets all values correctly."""
        config = _GlobalConfig()
        config.configure(
            api_url="http://localhost:8086",
            api_key="obx_live_test123",
            governance_timeout=45.0,
        )
        assert config.api_url == "http://localhost:8086"
        assert config.api_key == "obx_live_test123"
        assert config.governance_timeout == 45.0

    def test_configure_strips_trailing_slash(self):
        """Test configure() strips trailing slash from api_url."""
        config = _GlobalConfig()
        config.configure(
            api_url="http://localhost:8086/",
            api_key="obx_live_test123",
        )
        assert config.api_url == "http://localhost:8086"

    def test_configure_strips_multiple_trailing_slashes(self):
        """Test configure() strips multiple trailing slashes."""
        config = _GlobalConfig()
        config.configure(
            api_url="http://localhost:8086///",
            api_key="obx_live_test123",
        )
        # rstrip("/") removes all trailing slashes
        assert config.api_url == "http://localhost:8086"

    def test_configure_default_timeout(self):
        """Test configure() uses default timeout if not specified."""
        config = _GlobalConfig()
        config.configure(
            api_url="http://localhost:8086",
            api_key="obx_live_test123",
        )
        assert config.governance_timeout == 30.0

    def test_is_configured_false_when_empty(self):
        """Test is_configured() returns False when not configured."""
        config = _GlobalConfig()
        assert config.is_configured() is False

    def test_is_configured_false_when_only_url(self):
        """Test is_configured() returns False with only api_url."""
        config = _GlobalConfig()
        config.api_url = "http://localhost:8086"
        assert config.is_configured() is False

    def test_is_configured_false_when_only_key(self):
        """Test is_configured() returns False with only api_key."""
        config = _GlobalConfig()
        config.api_key = "obx_live_test123"
        assert config.is_configured() is False

    def test_is_configured_true_when_both_set(self):
        """Test is_configured() returns True when both are set."""
        config = _GlobalConfig()
        config.configure(
            api_url="http://localhost:8086",
            api_key="obx_live_test123",
        )
        assert config.is_configured() is True

    def test_reconfigure_updates_values(self):
        """Test configure() can be called multiple times."""
        config = _GlobalConfig()
        config.configure(
            api_url="http://localhost:8086",
            api_key="obx_live_first",
        )
        config.configure(
            api_url="http://production:8086",
            api_key="obx_live_second",
            governance_timeout=60.0,
        )
        assert config.api_url == "http://production:8086"
        assert config.api_key == "obx_live_second"
        assert config.governance_timeout == 60.0

    def test_repr_masks_long_api_key(self):
        """Test __repr__ masks API key longer than 8 characters."""
        config = _GlobalConfig()
        config.configure(
            api_url="https://api.openbox.ai",
            api_key="obx_live_abc123xyz789",
        )
        repr_str = repr(config)
        assert "obx_live_abc123xyz789" not in repr_str
        assert "obx_****" in repr_str
        assert "z789" in repr_str  # Last 4 chars visible
        assert "https://api.openbox.ai" in repr_str

    def test_repr_masks_short_api_key(self):
        """Test __repr__ masks API key 8 characters or shorter."""
        config = _GlobalConfig()
        config.configure(
            api_url="https://api.openbox.ai",
            api_key="short",
        )
        repr_str = repr(config)
        assert "short" not in repr_str
        assert "****" in repr_str
        assert "obx_****" not in repr_str  # Short key uses just ****

    def test_repr_handles_empty_api_key(self):
        """Test __repr__ handles empty API key."""
        config = _GlobalConfig()
        repr_str = repr(config)
        assert "api_key=''" in repr_str
        assert "api_url=''" in repr_str


# ═══════════════════════════════════════════════════════════════════════════════
# get_global_config() Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetGlobalConfig:
    """Test get_global_config() function."""

    def test_returns_singleton(self):
        """Test get_global_config() returns the same instance."""
        config1 = get_global_config()
        config2 = get_global_config()
        assert config1 is config2

    def test_returns_global_config_instance(self):
        """Test get_global_config() returns _GlobalConfig instance."""
        config = get_global_config()
        assert isinstance(config, _GlobalConfig)

    def test_modifications_persist(self):
        """Test that modifications to singleton persist."""
        config = get_global_config()
        original_url = config.api_url
        config.api_url = "http://test-persist:8086"

        config2 = get_global_config()
        assert config2.api_url == "http://test-persist:8086"

        # Restore original
        config.api_url = original_url


# ═══════════════════════════════════════════════════════════════════════════════
# initialize() Function Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestInitializeWithValidateFalse:
    """Test initialize() function with validate=False."""

    def test_configures_global_config(self):
        """Test initialize() configures the global config."""
        initialize(
            api_url="http://localhost:9000",
            api_key="obx_live_init_test",
            governance_timeout=25.0,
            validate=False,
        )
        config = get_global_config()
        assert config.api_url == "http://localhost:9000"
        assert config.api_key == "obx_live_init_test"
        assert config.governance_timeout == 25.0

    def test_strips_trailing_slash(self):
        """Test initialize() strips trailing slash from api_url."""
        initialize(
            api_url="http://localhost:9000/",
            api_key="obx_live_test",
            validate=False,
        )
        config = get_global_config()
        assert config.api_url == "http://localhost:9000"

    def test_default_governance_timeout(self):
        """Test initialize() uses default governance_timeout."""
        initialize(
            api_url="http://localhost:9000",
            api_key="obx_live_test",
            validate=False,
        )
        config = get_global_config()
        assert config.governance_timeout == 30.0

    def test_invalid_key_format_raises_exception(self):
        """Test initialize() raises OpenBoxAuthError for invalid key format."""
        with pytest.raises(OpenBoxAuthError) as exc_info:
            initialize(
                api_url="http://localhost:9000",
                api_key="invalid_key_format",
                validate=False,
            )

        assert "Invalid API key format" in str(exc_info.value)
        assert "obx_live_*" in str(exc_info.value)
        assert "obx_test_*" in str(exc_info.value)


class TestInitializeWithValidateTrue:
    """Test initialize() function with validate=True (mocked urllib)."""

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_validate_success(self, mock_request_class, mock_urlopen):
        """Test initialize() with successful server validation."""
        mock_response = MagicMock()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        initialize(
            api_url="http://localhost:9000",
            api_key="obx_live_valid_key",
            validate=True,
        )

        config = get_global_config()
        assert config.api_url == "http://localhost:9000"
        assert config.api_key == "obx_live_valid_key"

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_validate_auth_error_401(self, mock_request_class, mock_urlopen):
        """Test initialize() raises OpenBoxAuthError on 401."""
        mock_urlopen.side_effect = HTTPError(
            url="http://localhost:9000/api/v1/auth/validate",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=None,
        )

        with pytest.raises(OpenBoxAuthError) as exc_info:
            initialize(
                api_url="http://localhost:9000",
                api_key="obx_live_bad_key",
                validate=True,
            )

        assert "Invalid API key" in str(exc_info.value)

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_validate_auth_error_403(self, mock_request_class, mock_urlopen):
        """Test initialize() raises OpenBoxAuthError on 403."""
        mock_urlopen.side_effect = HTTPError(
            url="http://localhost:9000/api/v1/auth/validate",
            code=403,
            msg="Forbidden",
            hdrs={},
            fp=None,
        )

        with pytest.raises(OpenBoxAuthError) as exc_info:
            initialize(
                api_url="http://localhost:9000",
                api_key="obx_live_forbidden_key",
                validate=True,
            )

        assert "Invalid API key" in str(exc_info.value)

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_validate_network_error_http_500(self, mock_request_class, mock_urlopen):
        """Test initialize() raises OpenBoxNetworkError on HTTP 500."""
        mock_urlopen.side_effect = HTTPError(
            url="http://localhost:9000/api/v1/auth/validate",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=None,
        )

        with pytest.raises(OpenBoxNetworkError) as exc_info:
            initialize(
                api_url="http://localhost:9000",
                api_key="obx_live_test_key",
                validate=True,
            )

        assert "HTTP 500" in str(exc_info.value)

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_validate_network_error_url_error(self, mock_request_class, mock_urlopen):
        """Test initialize() raises OpenBoxNetworkError on URLError."""
        mock_urlopen.side_effect = URLError("Connection refused")

        with pytest.raises(OpenBoxNetworkError) as exc_info:
            initialize(
                api_url="http://localhost:9000",
                api_key="obx_live_test_key",
                validate=True,
            )

        assert "Cannot reach OpenBox Core" in str(exc_info.value)

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_validate_network_error_generic_exception(self, mock_request_class, mock_urlopen):
        """Test initialize() raises OpenBoxNetworkError on generic exception."""
        mock_urlopen.side_effect = Exception("Unknown error")

        with pytest.raises(OpenBoxNetworkError) as exc_info:
            initialize(
                api_url="http://localhost:9000",
                api_key="obx_live_test_key",
                validate=True,
            )

        assert "Cannot reach OpenBox Core" in str(exc_info.value)

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_validate_creates_correct_request(self, mock_request_class, mock_urlopen):
        """Test initialize() creates request with correct parameters."""
        mock_response = MagicMock()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        initialize(
            api_url="http://localhost:9000",
            api_key="obx_live_request_test",
            validate=True,
        )

        # Check Request was created correctly
        mock_request_class.assert_called_once()
        call_args = mock_request_class.call_args

        assert call_args[0][0] == "http://localhost:9000/api/v1/auth/validate"
        assert call_args[1]["headers"]["Authorization"] == "Bearer obx_live_request_test"
        assert call_args[1]["headers"]["Content-Type"] == "application/json"
        assert call_args[1]["method"] == "GET"

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_validate_uses_correct_timeout(self, mock_request_class, mock_urlopen):
        """Test initialize() passes correct timeout to urlopen."""
        mock_response = MagicMock()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        initialize(
            api_url="http://localhost:9000",
            api_key="obx_live_timeout_test",
            governance_timeout=45.0,
            validate=True,
        )

        # Check urlopen was called with correct timeout
        call_kwargs = mock_urlopen.call_args[1]
        assert call_kwargs["timeout"] == 45.0


# ═══════════════════════════════════════════════════════════════════════════════
# Exception Classes Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestExceptionClasses:
    """Test exception classes and their inheritance."""

    def test_openbox_config_error_is_exception(self):
        """Test OpenBoxConfigError inherits from Exception."""
        assert issubclass(OpenBoxConfigError, Exception)

    def test_openbox_auth_error_inherits_config_error(self):
        """Test OpenBoxAuthError inherits from OpenBoxConfigError."""
        assert issubclass(OpenBoxAuthError, OpenBoxConfigError)
        assert issubclass(OpenBoxAuthError, Exception)

    def test_openbox_network_error_inherits_config_error(self):
        """Test OpenBoxNetworkError inherits from OpenBoxConfigError."""
        assert issubclass(OpenBoxNetworkError, OpenBoxConfigError)
        assert issubclass(OpenBoxNetworkError, Exception)

    def test_openbox_config_error_can_be_raised(self):
        """Test OpenBoxConfigError can be raised and caught."""
        with pytest.raises(OpenBoxConfigError):
            raise OpenBoxConfigError("Config error message")

    def test_openbox_auth_error_can_be_raised(self):
        """Test OpenBoxAuthError can be raised and caught."""
        with pytest.raises(OpenBoxAuthError):
            raise OpenBoxAuthError("Auth error message")

    def test_openbox_network_error_can_be_raised(self):
        """Test OpenBoxNetworkError can be raised and caught."""
        with pytest.raises(OpenBoxNetworkError):
            raise OpenBoxNetworkError("Network error message")

    def test_auth_error_caught_as_config_error(self):
        """Test OpenBoxAuthError can be caught as OpenBoxConfigError."""
        with pytest.raises(OpenBoxConfigError):
            raise OpenBoxAuthError("Auth error")

    def test_network_error_caught_as_config_error(self):
        """Test OpenBoxNetworkError can be caught as OpenBoxConfigError."""
        with pytest.raises(OpenBoxConfigError):
            raise OpenBoxNetworkError("Network error")

    def test_exception_messages_preserved(self):
        """Test exception messages are preserved correctly."""
        config_error = OpenBoxConfigError("Config message")
        assert str(config_error) == "Config message"

        auth_error = OpenBoxAuthError("Auth message")
        assert str(auth_error) == "Auth message"

        network_error = OpenBoxNetworkError("Network message")
        assert str(network_error) == "Network message"

    def test_exception_with_empty_message(self):
        """Test exceptions with empty message."""
        error = OpenBoxConfigError("")
        assert str(error) == ""

    def test_exception_hierarchy_for_except_block(self):
        """Test that exception hierarchy works correctly in except blocks."""
        caught_type = None

        try:
            raise OpenBoxAuthError("Auth failed")
        except OpenBoxNetworkError:
            caught_type = "network"
        except OpenBoxAuthError:
            caught_type = "auth"
        except OpenBoxConfigError:
            caught_type = "config"

        assert caught_type == "auth"


# ═══════════════════════════════════════════════════════════════════════════════
# Edge Cases and Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_governance_config_with_empty_sets(self):
        """Test GovernanceConfig with explicitly empty sets."""
        config = GovernanceConfig(
            skip_workflow_types=set(),
            skip_signals=set(),
            skip_activity_types=set(),
            skip_hitl_activity_types=set(),
        )
        assert len(config.skip_workflow_types) == 0
        assert len(config.skip_signals) == 0
        assert len(config.skip_activity_types) == 0
        assert len(config.skip_hitl_activity_types) == 0

    def test_governance_config_with_large_sets(self):
        """Test GovernanceConfig with many items in sets."""
        many_workflows = {f"Workflow_{i}" for i in range(100)}
        config = GovernanceConfig(skip_workflow_types=many_workflows)
        assert len(config.skip_workflow_types) == 100

    def test_initialize_with_very_short_timeout(self):
        """Test initialize with very short timeout."""
        initialize(
            api_url="http://localhost:9000",
            api_key="obx_live_short_timeout",
            governance_timeout=0.001,
            validate=False,
        )
        config = get_global_config()
        assert config.governance_timeout == 0.001

    def test_initialize_with_very_long_timeout(self):
        """Test initialize with very long timeout."""
        initialize(
            api_url="http://localhost:9000",
            api_key="obx_live_long_timeout",
            governance_timeout=3600.0,
            validate=False,
        )
        config = get_global_config()
        assert config.governance_timeout == 3600.0

    def test_api_url_with_path(self):
        """Test api_url with path component."""
        initialize(
            api_url="http://localhost:9000/openbox/api",
            api_key="obx_live_with_path",
            validate=False,
        )
        config = get_global_config()
        assert config.api_url == "http://localhost:9000/openbox/api"

    def test_api_url_with_https(self):
        """Test api_url with HTTPS."""
        initialize(
            api_url="https://api.openbox.ai",
            api_key="obx_live_https",
            validate=False,
        )
        config = get_global_config()
        assert config.api_url == "https://api.openbox.ai"

    def test_api_key_with_underscores_in_suffix(self):
        """Test API key with underscores in suffix."""
        key = "obx_test_abc_def_ghi_123"
        assert _validate_api_key_format(key) is True

        initialize(
            api_url="http://localhost:9000",
            api_key=key,
            validate=False,
        )
        config = get_global_config()
        assert config.api_key == key


class TestValidateApiKeyWithServer:
    """Test _validate_api_key_with_server function directly."""

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_non_200_response_raises_auth_error(self, mock_request_class, mock_urlopen):
        """Test non-200 response (but not HTTPError) raises auth error."""
        from openbox.config import _validate_api_key_with_server

        mock_response = MagicMock()
        mock_response.getcode.return_value = 201  # Created, not 200 OK
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        with pytest.raises(OpenBoxAuthError):
            _validate_api_key_with_server(
                api_url="http://localhost:9000",
                api_key="obx_live_test",
                timeout=30.0,
            )
