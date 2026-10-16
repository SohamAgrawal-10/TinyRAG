
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


def is_out_of_scope(top_score: float | None, min_score: float = 0.30) -> bool:
    """Return whether a query should be rejected as out of scope."""
    return top_score is None or top_score < min_score


def main():
    index_dir = Path("data/index")
    dataset_path = Path("tests/fixtures/queries.json")
    out_of_scope_path = Path("tests/fixtures/out_of_scope_queries.json")
    k = 1
    min_score = 0.30


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

    with out_of_scope_path.open("r", encoding="utf-8") as file:
        out_of_scope_queries = json.load(file)

    rejected = 0

    for item in out_of_scope_queries:
        query = item["query"]
        query_vector = embedder.encode([query])[0]
        results = index.search(query_vector, top_k=1)

        top_score = results[0]["score"] if results else None
        was_rejected = is_out_of_scope(top_score, min_score)

        if was_rejected:
            rejected += 1

        print(f"\nOut-of-scope query: {query}")
        print(
            f"Top score: "
            f"{top_score if top_score is not None else 'No results'}"
        )
        print(f"Rejected: {was_rejected}")

    print("\n=== Out-of-Scope Evaluation ===")
    print(f"Queries evaluated: {len(out_of_scope_queries)}")
    print(f"Rejected: {rejected}")
    print(f"Rejection rate: {rejected / len(out_of_scope_queries):.3f}")


if __name__ == "__main__":
    main()
