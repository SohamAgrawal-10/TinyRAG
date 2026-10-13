
import numpy as np
import pytest

from tinyrag.embeddings import Embedder


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


def test_embedding_shape(embedder):
    vectors = embedder.encode([
        "Python functions are reusable blocks of code.",
        "Virtual environments isolate dependencies.",
    ])

    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32


def test_embeddings_are_normalized(embedder):
    vectors = embedder.encode([
        "Python functions are reusable blocks of code.",
    ])

    norms = np.linalg.norm(vectors, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-5)
