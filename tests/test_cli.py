
from typer.testing import CliRunner

from tinyrag.cli import app

runner = CliRunner()


def test_search_rejects_results_below_min_score(monkeypatch):
    from tinyrag import cli

    class FakeEmbedder:
        def encode(self, texts):
            return [[0.1] * 384]

    class FakeIndex:
        def search(self, query_vector, top_k):
            return [
                {
                    "score": 0.1384,
                    "source": "git_basics.md",
                    "start_line": 1,
                    "end_line": 2,
                    "text": "# Git Basics",
                    "chunk_id": "git_basics.md:0001",
                }
            ]

    monkeypatch.setattr(cli, "Embedder", FakeEmbedder)
    monkeypatch.setattr(cli, "load_index", lambda *args: FakeIndex())
    monkeypatch.setattr(
        cli.Path,
        "exists",
        lambda self: True,
    )

    result = runner.invoke(
        app,
        ["search", "What is the capital of Japan?", "--min-score", "0.30"],
    )

    assert result.exit_code == 0
    assert "No sufficiently relevant results found." in result.output


def test_search_keeps_results_above_min_score(monkeypatch):
    from tinyrag import cli

    class FakeEmbedder:
        def encode(self, texts):
            return [[0.1] * 384]

    class FakeIndex:
        def search(self, query_vector, top_k):
            return [
                {
                    "score": 0.5959,
                    "source": "git_basics.md",
                    "start_line": 13,
                    "end_line": 18,
                    "text": "## Branches",
                    "chunk_id": "git_basics.md:0013",
                }
            ]

    monkeypatch.setattr(cli, "Embedder", FakeEmbedder)
    monkeypatch.setattr(cli, "load_index", lambda *args: FakeIndex())
    monkeypatch.setattr(
        cli.Path,
        "exists",
        lambda self: True,
    )

    result = runner.invoke(
        app,
        [
            "search",
            "How can I work on a feature without changing the main branch?",
            "--min-score",
            "0.30",
        ],
    )

    assert result.exit_code == 0
    assert "git_basics.md" in result.output
    assert "0.5959" in result.output
