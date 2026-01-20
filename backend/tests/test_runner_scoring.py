"""Tests for worker scoring defaults."""
from app.benchmarks.base import ItemResult
from app.worker.runner import _apply_error_score_defaults


def test_error_default_sets_score_only():
    result = ItemResult(item_id="1", error="timeout")
    updated = _apply_error_score_defaults(result)
    assert updated.score == 0.0
    assert updated.is_correct is None


def test_error_default_preserves_existing_values():
    result = ItemResult(item_id="1", error="oops", score=0.25, is_correct=True)
    updated = _apply_error_score_defaults(result)
    assert updated.score == 0.25
    assert updated.is_correct is True


def test_no_error_no_change():
    result = ItemResult(item_id="1")
    updated = _apply_error_score_defaults(result)
    assert updated.score is None
    assert updated.is_correct is None
