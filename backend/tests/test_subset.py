"""Tests for deterministic subset sampling."""
import pytest

from app.benchmarks.base import BenchmarkAdapter


class MockAdapter(BenchmarkAdapter):
    """Mock adapter for testing."""

    def get_name(self) -> str:
        return "mock"

    def get_display_name(self) -> str:
        return "Mock"

    async def get_total_items(self) -> int:
        return 100

    async def enumerate_items(self):
        for i in range(100):
            yield str(i)

    async def evaluate_item(self, item_id: str):
        pass


@pytest.fixture
def adapter():
    """Create mock adapter."""
    return MockAdapter(None, "test-model")


def test_subset_determinism(adapter):
    """Test that subset sampling is deterministic."""
    items = [str(i) for i in range(100)]
    seed = "run123_benchmark1"

    subset1 = adapter.get_deterministic_subset(items, 10, seed)
    subset2 = adapter.get_deterministic_subset(items, 10, seed)

    assert subset1 == subset2
    assert len(subset1) == 10


def test_subset_different_seeds(adapter):
    """Test that different seeds produce different subsets."""
    items = [str(i) for i in range(100)]

    subset1 = adapter.get_deterministic_subset(items, 10, "run1_benchmark1")
    subset2 = adapter.get_deterministic_subset(items, 10, "run2_benchmark1")

    assert subset1 != subset2


def test_subset_percentages(adapter):
    """Test subset percentage calculations."""
    items = [str(i) for i in range(100)]
    seed = "test_seed"

    subset_1 = adapter.get_deterministic_subset(items, 1, seed)
    subset_5 = adapter.get_deterministic_subset(items, 5, seed)
    subset_10 = adapter.get_deterministic_subset(items, 10, seed)
    subset_25 = adapter.get_deterministic_subset(items, 25, seed)
    subset_50 = adapter.get_deterministic_subset(items, 50, seed)
    subset_100 = adapter.get_deterministic_subset(items, 100, seed)

    assert len(subset_1) == 1
    assert len(subset_5) == 5
    assert len(subset_10) == 10
    assert len(subset_25) == 25
    assert len(subset_50) == 50
    assert len(subset_100) == 100


def test_subset_small_dataset(adapter):
    """Test subset on small dataset."""
    items = ["a", "b", "c"]
    seed = "test"

    subset = adapter.get_deterministic_subset(items, 50, seed)
    assert len(subset) == 1  # 50% of 3 = 1.5, floor = 1

    subset = adapter.get_deterministic_subset(items, 10, seed)
    assert len(subset) == 1  # Minimum 1


def test_subset_preserves_item_identity(adapter):
    """Test that subset items are from original list."""
    items = [f"item_{i}" for i in range(50)]
    seed = "test"

    subset = adapter.get_deterministic_subset(items, 20, seed)

    for item in subset:
        assert item in items


def test_item_hash_consistency(adapter):
    """Test that item hash is consistent."""
    data1 = {"question": "What is 2+2?", "answer": "4"}
    data2 = {"question": "What is 2+2?", "answer": "4"}

    hash1 = adapter.compute_item_hash(data1)
    hash2 = adapter.compute_item_hash(data2)

    assert hash1 == hash2


def test_item_hash_different_data(adapter):
    """Test that different data produces different hash."""
    data1 = {"question": "What is 2+2?", "answer": "4"}
    data2 = {"question": "What is 3+3?", "answer": "6"}

    hash1 = adapter.compute_item_hash(data1)
    hash2 = adapter.compute_item_hash(data2)

    assert hash1 != hash2






























