
import numpy as np


class VectorIndex:
    def __init__(self):
        self.vectors = np.empty((0, 384), dtype=np.float32)
        self.metadata = []

    def add(self, vectors: np.ndarray, metadata: list[dict]):
        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim != 2 or vectors.shape[1] != 384:
            raise ValueError("Expected vectors with 384 dimensions")

        if len(vectors) != len(metadata):
            raise ValueError("Each vector must have matching metadata")

        if len(vectors) == 0:
            return

        if not np.isfinite(vectors).all():
            raise ValueError("Vectors must contain only finite values")

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms < 1e-12):
            raise ValueError("Zero vectors cannot be indexed")

        vectors = vectors / norms

        self.vectors = np.vstack([self.vectors, vectors])
        self.metadata.extend(metadata)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if len(self.vectors) == 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)

        if query.shape != (384,):
            raise ValueError("Query must have 384 dimensions")

        if not np.isfinite(query).all():
            raise ValueError("Query must contain only finite values")

        norm = np.linalg.norm(query)
        if norm < 1e-12:
            raise ValueError("Query cannot be a zero vector")

        query = query / norm
        scores = self.vectors @ query

        k = min(top_k, len(scores))
        indices = np.argsort(-scores, kind="stable")[:k]

        return [
            {
                **self.metadata[int(i)],
                "score": float(scores[i]),
            }
            for i in indices
        ]
