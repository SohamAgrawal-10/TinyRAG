
from pathlib import Path

from tinyrag.chunking import load_markdown


def main():
    docs_dir = Path("docs")
    chunks = load_markdown(docs_dir)

    print(f"\nFound {len(chunks)} chunks.\n")

    for chunk in chunks:
        print("=" * 50)
        print(f"ID: {chunk['chunk_id']}")
        print(f"Source: {chunk['source']}")
        print(f"Lines: {chunk['start_line']}-{chunk['end_line']}")
        print(chunk["text"])
        print()


if __name__ == "__main__":
    main()
