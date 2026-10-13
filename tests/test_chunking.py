
from pathlib import Path

from tinyrag.chunking import chunk_markdown, load_markdown


def test_chunk_markdown_preserves_source_and_lines(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()

    document = docs / "example.md"
    document.write_text(
        "# Example\n\nFirst paragraph.\nSecond paragraph.\n",
        encoding="utf-8",
    )

    chunks = chunk_markdown(document, docs, max_lines=3, overlap=1)

    assert len(chunks) == 2
    assert chunks[0]["source"] == "example.md"
    assert chunks[0]["start_line"] == 1
    assert chunks[0]["end_line"] == 3
    assert chunks[1]["start_line"] == 3


def test_load_markdown_finds_nested_files(tmp_path: Path):
    docs = tmp_path / "docs"
    nested = docs / "nested"
    nested.mkdir(parents=True)

    (nested / "example.md").write_text(
        "# Nested document\nSome content.",
        encoding="utf-8",
    )

    chunks = load_markdown(docs)

    assert len(chunks) == 1
    assert chunks[0]["source"] == "nested/example.md"
