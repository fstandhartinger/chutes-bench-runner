"""OOLONG scoring: live enum values and the released deterministic parser.

Both rules exist because the benchmark was scoring near-zero for reasons that
had nothing to do with the model:

  * ground truths arrive as lists, and `str([12])` is `"[12]"`, so a correct
    `12` never matched;
  * models prefix answers with a speaker label after reading a transcript, so
    `User: 88675` never matched `88675`.

The live dataset's enum spellings, date representation and comparison answer
templates are covered because each has previously taken a valid answer down a
different scoring branch than the released OOLONG-synth scorer.
"""
import pytest

from app.benchmarks.adapters.oolong import (
    OolongAdapter,
    _normalize_answer,
    _normalize_prediction,
    _test_shard_location,
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
        ("[datetime.date(2022, 5, 26)]", "2022-05-26"),
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
        ("Label: formal", "formal"),
        ("Date: 05/26/2022", "05/26/2022"),
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
        "more commonplace",  # not an OOLONG comparison relation
        "1988",
        "user data",  # "user" without a colon is content, not a label
    ],
)
def test_normalisation_preserves_content(raw):
    assert _normalize_prediction(raw) == raw


def test_normalisation_rescues_only_formatting():
    """The three cases observed in real runs, and what each must do."""
    # Formatting-only misses: rescued.
    assert not score_answer("88675", "User: 88675", "NUMERIC")[1]
    assert score_answer("88675", _normalize_prediction("User: 88675"), "NUMERIC")[1]
    assert score_answer("12", _normalize_prediction("Answer: 12"), "NUMERIC")[1]

    # The released OOLONG parser canonicalizes a comparison relation inside the
    # requested answer template; the opposite relation remains wrong.
    prediction = "Computers & Internet is less common than Family & Relationships"
    assert score_answer(
        "less common than", _normalize_prediction(prediction), "ANSWER_TYPE.COMPARISON"
    )[1]
    assert not score_answer(
        "more common than", _normalize_prediction(prediction), "ANSWER_TYPE.COMPARISON"
    )[1]


def test_numeric_partial_credit_survives_normalisation():
    """0.75^|y-y'| still applies; normalisation must not disturb it."""
    exact, _ = score_answer("10", _normalize_prediction("**10**"), "NUMERIC")
    near, _ = score_answer("10", _normalize_prediction("Answer: 11"), "NUMERIC")
    assert exact == 1.0
    assert 0.7 < near < 0.8


def test_live_dataset_answer_type_uses_numeric_partial_credit():
    score, correct = score_answer("10", "11", "ANSWER_TYPE.NUMERIC")

    assert score == pytest.approx(0.75)
    assert correct is False


def test_numeric_parser_rejects_decimal_format_like_released_scorer():
    score, correct = score_answer("10", "10.0", "ANSWER_TYPE.NUMERIC")

    assert score == 0.0
    assert correct is False


def test_non_numeric_exact_match_is_case_sensitive():
    score, correct = score_answer("True", "true", "ANSWER_TYPE.LABEL")

    assert score == 0.0
    assert correct is False


@pytest.mark.parametrize("prediction", ["05/26/2022", "2022-05-26", "May 26, 2022"])
def test_live_dataset_date_answers_are_parsed(prediction):
    score, correct = score_answer(
        "2022-05-26", prediction, "ANSWER_TYPE.DATE"
    )

    assert score == 1.0
    assert correct is True


@pytest.mark.asyncio
async def test_explicit_agentic_item_ids_are_not_replaced_by_hash_sampling():
    adapter = OolongAdapter(client=object(), model_slug="test")
    adapter.run_config = {"oolong_agentic": {"item_ids": ["1253", "1266"]}}
    adapter.get_name = lambda: "oolong_agentic"

    total, selected = await adapter.get_items_for_evaluation(1, "irrelevant", 3)

    assert total == 5200
    assert selected == ["1253", "1266"]


@pytest.mark.parametrize(
    ("item_index", "expected"),
    [
        (0, (0, 0)),
        (126, (0, 126)),
        (127, (1, 0)),
        (4317, (33, 126)),
        (4318, (34, 0)),
        (5199, (40, 125)),
    ],
)
def test_pinned_test_shard_location(item_index, expected):
    assert _test_shard_location(item_index) == expected


@pytest.mark.asyncio
async def test_preload_reads_explicit_ids_before_worker_selection(monkeypatch):
    adapter = OolongAdapter(client=object(), model_slug="test")
    adapter.run_config = {"oolong_agentic": {"item_ids": ["803", "804"]}}
    adapter.get_name = lambda: "oolong_agentic"
    called = []

    def targeted_preload():
        called.append(set(adapter._target_item_ids or set()))

    monkeypatch.setattr(adapter, "_preload_targeted_parquet_rows", targeted_preload)

    await adapter.preload()

    assert called == [{803, 804}]
    assert adapter._preloaded is True
