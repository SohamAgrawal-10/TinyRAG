
import numpy as np
import pytest

from tinyrag.index import VectorIndex


def test_search_returns_most_similar_vector():
    index = VectorIndex()

    vectors = np.array([
        [1, 0] + [0] * 382,
        [0, 1] + [0] * 382,
    ], dtype=np.float32)

    metadata = [
        {"chunk_id": "a", "text": "First passage"},
        {"chunk_id": "b", "text": "Second passage"},
    ]

    index.add(vectors, metadata)

    query = np.array([0, 1] + [0] * 382, dtype=np.float32)
    results = index.search(query, top_k=1)

    assert results[0]["chunk_id"] == "b"
    assert results[0]["score"] == pytest.approx(1.0)


def test_empty_index_returns_no_results():
    index = VectorIndex()

    query = np.ones(384, dtype=np.float32)

    assert index.search(query) == []


def test_mismatched_metadata_is_rejected():
    index = VectorIndex()
    vectors = np.ones((2, 384), dtype=np.float32)

    with pytest.raises(ValueError):
        index.add(vectors, [{"chunk_id": "a"}])
