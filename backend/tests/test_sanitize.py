from app.services.run_service import sanitize_payload, sanitize_text


def test_sanitize_text_strips_null_bytes() -> None:
    assert sanitize_text("hello") == "hello"
    assert sanitize_text("he\x00llo") == "hello"
    assert sanitize_text("\x00") == ""
    assert sanitize_text(None) is None


def test_sanitize_payload_strips_null_bytes_recursively() -> None:
    payload = {
        "stdout": "ok\x00done",
        "nested": ["a\x00b", {"inner": "\x00"}],
        "count": 2,
    }
    cleaned = sanitize_payload(payload)
    assert cleaned["stdout"] == "okdone"
    assert cleaned["nested"][0] == "ab"
    assert cleaned["nested"][1]["inner"] == ""
    assert cleaned["count"] == 2
