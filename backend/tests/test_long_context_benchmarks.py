"""Tests for long-context benchmark adapters (S-NIAH, OOLONG, OOLONG-Pairs).

These benchmarks are based on the RLM paper (arxiv:2512.24601v1).
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.benchmarks.adapters.s_niah import (
    SNIAHAdapter,
    _generate_haystack,
    _generate_needle_key,
    _generate_needle_value,
    _insert_needle_in_haystack,
)
from app.benchmarks.adapters.oolong import (
    OolongAdapter,
    _compute_numeric_score,
    _is_exact_match,
    _extract_answer,
)
from app.benchmarks.adapters.oolong_pairs import (
    OolongPairsAdapter,
    _compute_f1_score,
)


# =============================================================================
# S-NIAH Tests
# =============================================================================

class TestSNIAHHelpers:
    """Tests for S-NIAH helper functions."""

    def test_generate_haystack_deterministic(self):
        """Test haystack generation is deterministic."""
        haystack1 = _generate_haystack(seed=42, target_tokens=1000)
        haystack2 = _generate_haystack(seed=42, target_tokens=1000)
        assert haystack1 == haystack2

    def test_generate_haystack_different_seeds(self):
        """Test different seeds produce different haystacks."""
        haystack1 = _generate_haystack(seed=42, target_tokens=1000)
        haystack2 = _generate_haystack(seed=123, target_tokens=1000)
        assert haystack1 != haystack2

    def test_generate_haystack_length(self):
        """Test haystack is approximately target length."""
        haystack = _generate_haystack(seed=42, target_tokens=1000)
        # Rough estimate: 1 token ≈ 4 characters
        estimated_tokens = len(haystack) // 4
        # Should be at least target_tokens
        assert estimated_tokens >= 1000

    def test_generate_needle_key_words(self):
        """Test needle key generation with words."""
        key = _generate_needle_key(seed=42, key_type="words")
        assert " " in key  # Should be "adjective noun"
        assert len(key) > 5

    def test_generate_needle_key_numbers(self):
        """Test needle key generation with numbers."""
        key = _generate_needle_key(seed=42, key_type="numbers")
        assert key.isdigit()
        assert len(key) == 7

    def test_generate_needle_key_uuids(self):
        """Test needle key generation with UUIDs."""
        key = _generate_needle_key(seed=42, key_type="uuids")
        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)

    def test_generate_needle_value_deterministic(self):
        """Test needle value generation is deterministic."""
        val1 = _generate_needle_value(seed=42)
        val2 = _generate_needle_value(seed=42)
        assert val1 == val2

    def test_insert_needle_at_position(self):
        """Test needle insertion at different positions."""
        haystack = "Para1.\n\nPara2.\n\nPara3.\n\nPara4.\n\nPara5."
        needle_key = "secret code"
        needle_value = "blue dragon"

        # Insert at beginning
        result_start = _insert_needle_in_haystack(haystack, needle_key, needle_value, 0.0, 42)
        assert "The secret code is: blue dragon." in result_start
        assert result_start.index("blue dragon") < result_start.index("Para2")

        # Insert at end
        result_end = _insert_needle_in_haystack(haystack, needle_key, needle_value, 1.0, 42)
        assert "The secret code is: blue dragon." in result_end
        assert result_end.index("blue dragon") > result_end.index("Para4")

        # Insert in middle
        result_mid = _insert_needle_in_haystack(haystack, needle_key, needle_value, 0.5, 42)
        assert "The secret code is: blue dragon." in result_mid


class TestSNIAHAdapter:
    """Tests for S-NIAH adapter."""

    @pytest.fixture
    def mock_client(self):
        """Create mock inference client."""
        client = MagicMock()
        client.get_completion_text = AsyncMock(
            return_value=("blue dragon", {"usage": {"prompt_tokens": 100, "completion_tokens": 10}})
        )
        return client

    @pytest.fixture
    def adapter(self, mock_client):
        """Create S-NIAH adapter with small context sizes for testing."""
        return SNIAHAdapter(
            mock_client,
            "test-model",
            context_sizes=[1024, 2048],  # Small for testing
        )

    @pytest.mark.asyncio
    async def test_adapter_name(self, adapter):
        """Test adapter name."""
        assert adapter.get_name() == "s_niah"
        assert adapter.get_display_name() == "S-NIAH (Needle-in-Haystack)"

    @pytest.mark.asyncio
    async def test_adapter_supports_parallel(self, adapter):
        """Test parallel support."""
        assert adapter.supports_parallel_items() is True

    @pytest.mark.asyncio
    async def test_preload_generates_items(self, adapter):
        """Test preload generates expected number of items."""
        await adapter.preload()
        # 2 context sizes × 10 tasks = 20 items
        assert len(adapter._items) == 20

    @pytest.mark.asyncio
    async def test_enumerate_items(self, adapter):
        """Test enumerate_items yields all item IDs."""
        await adapter.preload()
        items = []
        async for item_id in adapter.enumerate_items():
            items.append(item_id)
        assert len(items) == 20
        assert items[0] == "0"

    @pytest.mark.asyncio
    async def test_items_have_required_fields(self, adapter):
        """Test generated items have all required fields."""
        await adapter.preload()
        for item in adapter._items:
            assert "id" in item
            assert "context_size" in item
            assert "needle_key" in item
            assert "needle_value" in item
            assert "needle_position" in item
            assert "context" in item
            # Verify needle is actually in the context
            assert item["needle_value"] in item["context"]

    @pytest.mark.asyncio
    async def test_evaluate_item_correct_answer(self, adapter):
        """Test evaluation with correct answer."""
        await adapter.preload()
        item = adapter._items[0]

        # Mock client to return the correct needle value
        adapter.client.get_completion_text = AsyncMock(
            return_value=(item["needle_value"], {"usage": {"prompt_tokens": 100, "completion_tokens": 10}})
        )

        result = await adapter.evaluate_item("0")
        assert result.is_correct is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_item_wrong_answer(self, adapter):
        """Test evaluation with wrong answer."""
        await adapter.preload()

        adapter.client.get_completion_text = AsyncMock(
            return_value=("wrong answer", {"usage": {"prompt_tokens": 100, "completion_tokens": 10}})
        )

        result = await adapter.evaluate_item("0")
        assert result.is_correct is False
        assert result.score == 0.0


# =============================================================================
# OOLONG Tests
# =============================================================================

class TestOolongHelpers:
    """Tests for OOLONG helper functions."""

    def test_compute_numeric_score_exact(self):
        """Test numeric scoring with exact match."""
        score = _compute_numeric_score("42", "42")
        assert score == 1.0  # 0.75^0 = 1

    def test_compute_numeric_score_off_by_one(self):
        """Test numeric scoring off by one."""
        score = _compute_numeric_score("42", "43")
        assert score == 0.75  # 0.75^1

    def test_compute_numeric_score_off_by_two(self):
        """Test numeric scoring off by two."""
        score = _compute_numeric_score("42", "44")
        assert abs(score - 0.5625) < 0.001  # 0.75^2

    def test_compute_numeric_score_invalid(self):
        """Test numeric scoring with invalid input."""
        score = _compute_numeric_score("42", "not a number")
        assert score == 0.0

    def test_is_exact_match_true(self):
        """Test exact match detection - positive case."""
        assert _is_exact_match("spam", "spam") is True
        assert _is_exact_match("SPAM", "spam") is True
        assert _is_exact_match("  spam  ", "spam") is True

    def test_is_exact_match_false(self):
        """Test exact match detection - negative case."""
        assert _is_exact_match("spam", "ham") is False
        assert _is_exact_match("spam", "spammy") is False

    def test_extract_answer_explicit(self):
        """Test answer extraction from explicit format."""
        response = "Let me think about this.\n\nAnswer: spam"
        assert _extract_answer(response) == "spam"

    def test_extract_answer_the_answer_is(self):
        """Test answer extraction from 'the answer is' format."""
        response = "Based on the data, the answer is ham."
        assert _extract_answer(response) == "ham"

    def test_extract_answer_with_thinking(self):
        """Test answer extraction removes thinking blocks."""
        response = "<think>Let me analyze...</think>Answer: 42"
        assert _extract_answer(response) == "42"

    def test_extract_answer_last_line(self):
        """Test answer extraction from last line."""
        response = "After analysis...\nThe count is:\n5"
        assert _extract_answer(response) == "5"


class TestOolongAdapter:
    """Tests for OOLONG adapter."""

    @pytest.fixture
    def mock_client(self):
        """Create mock inference client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def adapter(self, mock_client):
        """Create OOLONG adapter."""
        return OolongAdapter(mock_client, "test-model")

    def test_adapter_name(self, adapter):
        """Test adapter name."""
        assert adapter.get_name() == "oolong"
        assert adapter.get_display_name() == "OOLONG"

    def test_adapter_supports_parallel(self, adapter):
        """Test parallel support."""
        assert adapter.supports_parallel_items() is True


# =============================================================================
# OOLONG-Pairs Tests
# =============================================================================

class TestOolongPairsHelpers:
    """Tests for OOLONG-Pairs helper functions."""

    def test_compute_f1_score_perfect(self):
        """Test F1 scoring with perfect match."""
        precision, recall, f1 = _compute_f1_score(
            "the quick brown fox",
            "the quick brown fox"
        )
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_compute_f1_score_partial(self):
        """Test F1 scoring with partial overlap."""
        precision, recall, f1 = _compute_f1_score(
            "the quick brown fox",
            "the quick red fox"  # 3/4 overlap
        )
        assert precision == 0.75  # 3 correct out of 4 predicted
        assert recall == 0.75  # 3 correct out of 4 expected
        assert f1 == 0.75

    def test_compute_f1_score_no_overlap(self):
        """Test F1 scoring with no overlap."""
        precision, recall, f1 = _compute_f1_score(
            "the quick brown fox",
            "a lazy old cat"
        )
        assert f1 == 0.0

    def test_compute_f1_score_empty(self):
        """Test F1 scoring with empty strings."""
        precision, recall, f1 = _compute_f1_score("", "test")
        assert f1 == 0.0

        precision, recall, f1 = _compute_f1_score("test", "")
        assert f1 == 0.0

    def test_compute_f1_score_case_insensitive(self):
        """Test F1 scoring is case insensitive."""
        precision, recall, f1 = _compute_f1_score(
            "The Quick Brown Fox",
            "the quick brown fox"
        )
        assert f1 == 1.0


class TestOolongPairsAdapter:
    """Tests for OOLONG-Pairs adapter."""

    @pytest.fixture
    def mock_client(self):
        """Create mock inference client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def adapter(self, mock_client):
        """Create OOLONG-Pairs adapter."""
        return OolongPairsAdapter(mock_client, "test-model")

    def test_adapter_name(self, adapter):
        """Test adapter name."""
        assert adapter.get_name() == "oolong_pairs"
        assert adapter.get_display_name() == "OOLONG-Pairs"

    def test_adapter_supports_parallel(self, adapter):
        """Test parallel support."""
        assert adapter.supports_parallel_items() is True

    def test_adapter_timeout(self, adapter):
        """Test longer timeout for quadratic complexity."""
        assert adapter.get_item_timeout_seconds() == 300


# =============================================================================
# Registry Tests
# =============================================================================

class TestBenchmarkRegistration:
    """Tests for benchmark adapter registration."""

    def test_s_niah_registered(self):
        """Test S-NIAH adapter is registered."""
        from app.benchmarks.registry import get_all_adapters
        adapters = get_all_adapters()
        assert "s_niah" in adapters

    def test_oolong_registered(self):
        """Test OOLONG adapter is registered."""
        from app.benchmarks.registry import get_all_adapters
        adapters = get_all_adapters()
        assert "oolong" in adapters

    def test_oolong_pairs_registered(self):
        """Test OOLONG-Pairs adapter is registered."""
        from app.benchmarks.registry import get_all_adapters
        adapters = get_all_adapters()
        assert "oolong_pairs" in adapters

    def test_get_adapter_s_niah(self):
        """Test getting S-NIAH adapter instance."""
        from app.benchmarks.registry import get_adapter
        mock_client = MagicMock()
        adapter = get_adapter("s_niah", mock_client, "test-model")
        assert adapter is not None
        assert adapter.get_name() == "s_niah"

    def test_get_adapter_oolong(self):
        """Test getting OOLONG adapter instance."""
        from app.benchmarks.registry import get_adapter
        mock_client = MagicMock()
        adapter = get_adapter("oolong", mock_client, "test-model")
        assert adapter is not None
        assert adapter.get_name() == "oolong"

    def test_get_adapter_oolong_pairs(self):
        """Test getting OOLONG-Pairs adapter instance."""
        from app.benchmarks.registry import get_adapter
        mock_client = MagicMock()
        adapter = get_adapter("oolong_pairs", mock_client, "test-model")
        assert adapter is not None
        assert adapter.get_name() == "oolong_pairs"
