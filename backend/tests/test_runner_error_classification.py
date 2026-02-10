"""Tests for worker retry/fatal error classification."""

from app.worker.runner import _is_fatal_item_error, _is_retryable_item_error


def test_sandy_all_upstreams_failed_is_fatal_and_retryable() -> None:
    error = 'HTTP 503: {"detail":"All upstreams failed to create sandbox: status 503"} (after 5 attempts)'
    assert _is_fatal_item_error(error) is True
    assert _is_retryable_item_error(error) is True


def test_sandy_api_key_not_configured_is_fatal_and_not_retryable() -> None:
    error = "Sandy API key is not configured"
    assert _is_fatal_item_error(error) is True
    assert _is_retryable_item_error(error) is False
