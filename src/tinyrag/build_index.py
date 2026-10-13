
from pathlib import Path

from tinyrag.chunking import load_markdown
from tinyrag.embeddings import Embedder
from tinyrag.index import VectorIndex
from tinyrag.storage import save_index


def main():
    docs_dir = Path("docs")
    index_dir = Path("data/index")

    chunks = load_markdown(docs_dir)

    if not chunks:
        print("No Markdown chunks found.")
        return

    print(f"Found {len(chunks)} chunks. Generating embeddings...")

    embedder = Embedder()
    vectors = embedder.encode([chunk["text"] for chunk in chunks])

    index = VectorIndex()
    index.add(vectors, chunks)

    save_index(index, index_dir)

    print(f"Index saved to: {index_dir}")
    print(f"Indexed chunks: {len(chunks)}")


if __name__ == "__main__":
    main()
