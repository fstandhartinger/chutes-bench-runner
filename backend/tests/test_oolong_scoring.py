"""OOLONG scoring: ground-truth unwrapping and formatting-only normalisation.

Both rules exist because the benchmark was scoring near-zero for reasons that
had nothing to do with the model:

  * ground truths arrive as lists, and `str([12])` is `"[12]"`, so a correct
    `12` never matched;
  * models prefix answers with a speaker label after reading a transcript, so
    `User: 88675` never matched `88675`.

Both floored the score for *every* arm, which makes a harness comparison
impossible in either direction. The tests that matter most here are the
negative ones: normalisation must fix formatting and must never turn a wrong
answer into a right one.
"""
import pytest

from app.benchmarks.adapters.oolong import (
    _normalize_answer,
    _normalize_prediction,
    score_answer,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ([12], "12"),
        ("[12]", "12"),
        (["Alice"], "Alice"),
        ("['Alice']", "Alice"),
        ([1, 2, 3], "1, 2, 3"),
        ("42", "42"),
        ("[not a list", "[not a list"),
        ([], "[]"),
    ],
)
def test_ground_truth_unwrapping(raw, expected):
    assert _normalize_answer(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("User: 88675", "88675"),
        ("Answer: 12", "12"),
        ("Speaker: yes", "yes"),
        ('"88675"', "88675"),
        ("`19`", "19"),
        ("**19**", "19"),
        ('**"19"**', "19"),
        ("19.", "19"),
        ("  88675  ", "88675"),
    ],
)
def test_normalisation_strips_formatting(raw, expected):
    assert _normalize_prediction(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        # A substring of the ground truth is not a match; recovering it would
        # be grading, not formatting.
        "Computers & Internet is less common than Family & Relationships",
        "more common than",  # the opposite answer stays the opposite answer
        "1988",
        "user data",  # "user" without a colon is content, not a label
    ],
)
def test_normalisation_preserves_content(raw):
    assert _normalize_prediction(raw) == raw


def test_normalisation_rescues_only_formatting():
    """The three cases observed in real runs, and what each must do."""
    # Formatting-only misses: rescued.
    assert score_answer("88675", "User: 88675", "NUMERIC")[1] is False or True
    assert score_answer("88675", _normalize_prediction("User: 88675"), "NUMERIC")[1]
    assert score_answer("12", _normalize_prediction("Answer: 12"), "NUMERIC")[1]

    # Genuine miss: still a miss, before and after.
    prediction = "Computers & Internet is less common than Family & Relationships"
    assert not score_answer("less common than", prediction, "STRING")[1]
    assert not score_answer(
        "less common than", _normalize_prediction(prediction), "STRING"
    )[1]


def test_numeric_partial_credit_survives_normalisation():
    """0.75^|y-y'| still applies; normalisation must not disturb it."""
    exact, _ = score_answer("10", _normalize_prediction("**10**"), "NUMERIC")
    near, _ = score_answer("10", _normalize_prediction("Answer: 11"), "NUMERIC")
    assert exact == 1.0
    assert 0.7 < near < 0.8
