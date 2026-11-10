
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from tinyrag.generator import Generator
from tinyrag.chunking import load_markdown
from tinyrag.embeddings import Embedder
from tinyrag.index import VectorIndex
from tinyrag.storage import load_index, save_index

app = typer.Typer(help="TinyRAG: local semantic search for Markdown docs.")
console = Console()

INDEX_DIR = Path("data/index")


@app.command()
def index(
    docs_dir: Path = typer.Argument(Path("docs"), exists=True, file_okay=False),
):
    """Build and save an index from Markdown files."""
    chunks = load_markdown(docs_dir)

    if not chunks:
        console.print("[yellow]No Markdown chunks found.[/yellow]")
        raise typer.Exit(code=1)

    console.print(f"Embedding {len(chunks)} chunks...")
    embedder = Embedder()
    vectors = embedder.encode([chunk["text"] for chunk in chunks])

    vector_index = VectorIndex()
    vector_index.add(vectors, chunks)
    save_index(vector_index, INDEX_DIR)

    console.print(f"[green]Indexed {len(chunks)} chunks.[/green]")
    console.print(f"Saved index to {INDEX_DIR}")



@app.command()
def search(
    query: str = typer.Argument(..., help="Question to search for."),
    top_k: int = typer.Option(3, "--top-k", min=1, help="Number of results."),
    min_score: float | None = typer.Option(
        None,
        "--min-score",
        min=-1.0,
        max=1.0,
        help="Optional minimum cosine similarity score.",
    ),
):
    """Search the saved index."""
    if not (INDEX_DIR / "vectors.npy").exists():
        console.print("[red]No index found. Run 'tinyrag index' first.[/red]")
        raise typer.Exit(code=1)

    vector_index = load_index(INDEX_DIR, VectorIndex)
    embedder = Embedder()
    query_vector = embedder.encode([query])[0]
    results = vector_index.search(query_vector, top_k=top_k)

    if min_score is not None:
        results = [
            result for result in results
            if result["score"] >= min_score
        ]

    if not results:
        console.print(
            "[yellow]No sufficiently relevant results found.[/yellow]"
        )
        raise typer.Exit(code=0)

    table = Table(title="TinyRAG Search Results")
    table.add_column("Rank", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Source")
    table.add_column("Passage")

    for rank, result in enumerate(results, start=1):
        table.add_row(
            str(rank),
            f"{result['score']:.4f}",
            f"{result['source']}:{result['start_line']}-{result['end_line']}",
            result["text"][:200].replace("\n", " "),
        )

    console.print(table)

@app.command()
def ask(
    query: str = typer.Argument(..., help="Question to answer."),
    top_k: int = typer.Option(
        3,
        "--top-k",
        min=1,
        help="Number of context chunks.",
    ),
    min_score: float = typer.Option(
        0.45,
        "--min-score",
        min=-1.0,
        max=1.0,
        help="Minimum similarity required for retrieval.",
    ),
):
    """Answer a question using retrieved document context."""
    if not (INDEX_DIR / "vectors.npy").exists():
        console.print("[red]No index found. Run 'tinyrag index' first.[/red]")
        raise typer.Exit(code=1)

    vector_index = load_index(INDEX_DIR, VectorIndex)
    embedder = Embedder()
    generator = Generator()

    query_vector = embedder.encode([query])[0]
    results = vector_index.search(query_vector, top_k=1)

    results = [
        result for result in results
        if result["score"] >= min_score
    ]

    if not results:
        console.print(
            "[yellow]No sufficiently relevant context found.[/yellow]"
        )
        raise typer.Exit(code=0)

    console.print("[bold]Generating answer...[/bold]\n")

    answer = generator.generate(query, results)

    console.print("[bold green]Answer[/bold green]")
    console.print(answer)


if __name__ == "__main__":
    app()
