
from pathlib import Path

from tinyrag.chunking import load_markdown
from tinyrag.embeddings import Embedder
from tinyrag.index import VectorIndex


def main():
    docs_dir = Path("docs")

    # 1. Read and chunk the Markdown documents.
    chunks = load_markdown(docs_dir)

    if not chunks:
        print("No Markdown chunks found in the docs directory.")
        return

    print(f"Loaded {len(chunks)} chunks.")

    # 2. Convert every chunk into a vector.
    embedder = Embedder()
    vectors = embedder.encode([chunk["text"] for chunk in chunks])

    # 3. Build the vector index.
    index = VectorIndex()
    index.add(vectors, chunks)

    # 4. Ask a question.
    question = input("\nAsk a question about your docs: ").strip()

    if not question:
        print("Please enter a question.")
        return

    # 5. Retrieve the most similar chunks.
    query_vector = embedder.encode([question])[0]
    results = index.search(query_vector, top_k=3)

    print("\nMost relevant passages:\n")

    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result['source']}")
        print(
            f"   Lines {result['start_line']}-{result['end_line']}"
        )
        print(f"   Similarity: {result['score']:.4f}")
        print(f"   {result['text']}\n")


if __name__ == "__main__":
    main()
