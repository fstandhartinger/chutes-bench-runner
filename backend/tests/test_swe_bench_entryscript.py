"""Tests for SWE-Bench Pro harness script generation."""

from app.benchmarks.adapters.swe_bench import SWEBenchProAdapter


def test_entryscript_uses_only_last_line_of_before_repo_set_cmd(monkeypatch) -> None:
    # Avoid network calls by stubbing dockerfile downloads.
    adapter = SWEBenchProAdapter.__new__(SWEBenchProAdapter)
    monkeypatch.setattr(adapter, "_download_dockerfile", lambda *_args, **_kwargs: "ENV FOO=bar\n")

    sample = {
        "instance_id": "instance_dummy",
        "base_commit": "deadbeef",
        "selected_test_files_to_run": '["tests/test_foo.py"]',
        "before_repo_set_cmd": "\n".join(
            [
                "git reset --hard deadbeef",
                "git clean -fd",
                "git checkout deadbeef",
                "git checkout cafebabe -- tests/test_foo.py",
            ]
        ),
    }

    entry = SWEBenchProAdapter._create_entryscript(adapter, sample)

    # The harness itself resets/checks out the base commit once.
    assert entry.count("git reset --hard deadbeef") == 1
    assert entry.count("git checkout deadbeef") == 1

    # Only the final checkout line from before_repo_set_cmd should be present.
    assert "git clean -fd" not in entry
    assert entry.count("git checkout cafebabe -- tests/test_foo.py") == 1
