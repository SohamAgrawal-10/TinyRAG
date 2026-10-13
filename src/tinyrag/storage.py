
import json
from pathlib import Path

import numpy as np


def save_index(index, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)

    np.save(directory / "vectors.npy", index.vectors)

    with (directory / "metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(index.metadata, file, ensure_ascii=False, indent=2)


def load_index(directory: Path, index_class):
    vectors_path = directory / "vectors.npy"
    metadata_path = directory / "metadata.json"

    vectors = np.load(vectors_path, allow_pickle=False)

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if vectors.ndim != 2 or vectors.shape[1] != 384:
        raise ValueError("Invalid vector index dimensions")

    if len(vectors) != len(metadata):
        raise ValueError("Vector and metadata counts do not match")

    index = index_class()
    index.vectors = vectors.astype(np.float32, copy=False)
    index.metadata = metadata

    return index
