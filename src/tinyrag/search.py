
from pathlib import Path

from tinyrag.embeddings import Embedder
from tinyrag.index import VectorIndex
from tinyrag.storage import load_index


def main():
    index_dir = Path("data/index")

    if not index_dir.exists():
        print("No saved index found. Run build_index.py first.")
        return

    index = load_index(index_dir, VectorIndex)
    embedder = Embedder()

    print(f"Loaded {len(index.metadata)} indexed chunks.")

    while True:
        question = input("\nAsk a question (or type 'exit'): ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        query_vector = embedder.encode([question])[0]
        results = index.search(query_vector, top_k=3)

        print("\nMost relevant passages:\n")

        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. {result['source']} "
                f"(lines {result['start_line']}-{result['end_line']})"
            )
            print(f"   Similarity: {result['score']:.4f}")
            print(f"   {result['text']}\n")


if __name__ == "__main__":
    main()
