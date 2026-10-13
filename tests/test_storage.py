
import numpy as np

from tinyrag.index import VectorIndex
from tinyrag.storage import save_index, load_index


def test_save_and_load_index(tmp_path):
    index = VectorIndex()

    vectors = np.zeros((2, 384), dtype=np.float32)
    vectors[0, 0] = 1.0
    vectors[1, 1] = 1.0

    metadata = [
        {
            "chunk_id": "a",
            "source": "example.md",
            "text": "First passage",
        },
        {
            "chunk_id": "b",
            "source": "example.md",
            "text": "Second passage",
        },
    ]

    index.add(vectors, metadata)

    save_index(index, tmp_path)
    loaded = load_index(tmp_path, VectorIndex)

    assert loaded.vectors.shape == (2, 384)
    assert loaded.metadata == metadata
    assert np.allclose(loaded.vectors, index.vectors)


def test_empty_index_can_be_saved_and_loaded(tmp_path):
    index = VectorIndex()

    save_index(index, tmp_path)
    loaded = load_index(tmp_path, VectorIndex)

    assert loaded.vectors.shape == (0, 384)
    assert loaded.metadata == []
