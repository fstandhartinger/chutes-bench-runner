"""Tests for worker retry/fatal error classification."""

from app.worker.runner import (
    ITEM_TIMEOUT_EXCLUSION_REASON,
    _accuracy_excluding_infrastructure,
    _apply_error_score_defaults,
    _is_fatal_item_error,
    _is_retryable_item_error,
    _item_timeout_result,
)


def test_sandy_all_upstreams_failed_is_fatal_and_retryable() -> None:
    error = 'HTTP 503: {"detail":"All upstreams failed to create sandbox: status 503"} (after 5 attempts)'
    assert _is_fatal_item_error(error) is True
    assert _is_retryable_item_error(error) is True


def test_sandy_api_key_not_configured_is_fatal_and_not_retryable() -> None:
    error = "Sandy API key is not configured"
    assert _is_fatal_item_error(error) is True
    assert _is_retryable_item_error(error) is False


def test_disabled_chute_is_fatal_and_not_retryable() -> None:
    error = 'HTTP 503 from Chutes: {"detail":"This chute is currently disabled."}'
    assert _is_fatal_item_error(error) is True
    assert _is_retryable_item_error(error) is False


def test_sandy_connection_failures_are_fatal_and_retryable() -> None:
    error = "All connection attempts failed"
    assert _is_fatal_item_error(error) is True
    assert _is_retryable_item_error(error) is True


def test_runner_item_timeout_is_excluded_and_not_retried() -> None:
    result = _apply_error_score_defaults(_item_timeout_result("hard-item", 1200))

    assert result.error == "Item evaluation timed out after 1200s"
    assert result.metadata is not None
    assert result.metadata["exclusion_reason"] == ITEM_TIMEOUT_EXCLUSION_REASON
    assert result.score is None
    assert _is_retryable_item_error(result.error) is False
    assert _accuracy_excluding_infrastructure(1, 3, 1) == 0.5
