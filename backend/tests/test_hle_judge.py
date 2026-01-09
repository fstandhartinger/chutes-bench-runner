"""Tests for HLE judge parsing."""

from app.benchmarks.adapters.hle import HLEAdapter


def test_hle_judge_parses_json():
    adapter = HLEAdapter(None, "mock")
    payload = adapter._extract_judge_payload(
        '{"extracted_final_answer":"42","reasoning":"ok","correct":"yes","confidence":90}'
    )
    assert payload["extracted_final_answer"] == "42"
    assert payload["correct"] == "yes"
    assert payload["confidence"] == 90


def test_hle_judge_parses_text_fields():
    adapter = HLEAdapter(None, "mock")
    payload = adapter._extract_judge_payload(
        "extracted_final_answer: Rxf3\nreasoning: matches\ncorrect: no\nconfidence: 12"
    )
    assert payload["extracted_final_answer"] == "Rxf3"
    assert payload["correct"] == "no"
    assert payload["confidence"] == 12


def test_hle_judge_empty_response():
    adapter = HLEAdapter(None, "mock")
    payload = adapter._extract_judge_payload("")
    assert payload == {}
