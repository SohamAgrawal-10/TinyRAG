
import json
from pathlib import Path

from tinyrag.embeddings import Embedder
from tinyrag.evaluation import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from tinyrag.index import VectorIndex
from tinyrag.storage import load_index


def main():
    index_dir = Path("data/index")
    dataset_path = Path("tests/fixtures/queries.json")
    k = 1

    if not (index_dir / "vectors.npy").exists():
        print("No saved index found. Run 'tinyrag index' first.")
        return

    with dataset_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    index = load_index(index_dir, VectorIndex)
    embedder = Embedder()

    precisions = []
    recalls = []
    reciprocal_ranks = []

    for item in dataset:
        query = item["query"]
        relevant = item["relevant_chunks"]

        query_vector = embedder.encode([query])[0]
        results = index.search(query_vector, top_k=k)
        retrieved = [result["chunk_id"] for result in results]

        precision = precision_at_k(retrieved, relevant, k)
        recall = recall_at_k(retrieved, relevant, k)
        rr = reciprocal_rank(retrieved, relevant)

        precisions.append(precision)
        recalls.append(recall)
        reciprocal_ranks.append(rr)

        print(f"\nQuery: {query}")
        print(f"Retrieved: {retrieved}")
        print(f"Precision@{k}: {precision:.3f}")
        print(f"Recall@{k}: {recall:.3f}")
        print(f"Reciprocal rank: {rr:.3f}")

    print("\n=== Aggregate Results ===")
    print(f"Queries evaluated: {len(dataset)}")
    print(f"Mean Precision@{k}: {sum(precisions) / len(precisions):.3f}")
    print(f"Mean Recall@{k}: {sum(recalls) / len(recalls):.3f}")
    print(f"MRR: {sum(reciprocal_ranks) / len(reciprocal_ranks):.3f}")


if __name__ == "__main__":
    main()
