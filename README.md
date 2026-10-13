# TinyRAG

A lightweight, local Retrieval-Augmented Generation (RAG) project built with Python. TinyRAG currently implements document chunking, semantic search using sentence embeddings, persistent vector indexing, and retrieval evaluation.

The project is being developed incrementally to explore how retrieval systems work internally, from processing documents to ranking relevant passages.

## Features

- **Markdown ingestion:** Load Markdown documents from a directory.
- **Heading-aware chunking:** Split documents into manageable sections while preserving headings and source line information.
- **Semantic embeddings:** Generate text embeddings locally using `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector search:** Rank document chunks using cosine similarity.
- **Persistent indexing:** Save and reload vectors and metadata locally.
- **Command-line interface:** Index documents and search them from the terminal.
- **Retrieval evaluation:** Measure Precision@k, Recall@k, and Mean Reciprocal Rank (MRR).
- **Automated tests:** Test core chunking, embedding, indexing, storage, and evaluation functionality.

## Tech Stack

- Python
- Sentence Transformers
- NumPy
- Typer
- Rich
- pytest
- Ruff

## Architecture

```text
Markdown Documents
       |
       v
Document Loading & Chunking
       |
       v
Text Embeddings
       |
       v
Vector Index + Metadata
       |
       v
Semantic Search
       |
       v
Ranked Document Chunks
       |
       v
Retrieval Evaluation
```

## Getting Started

### Prerequisites

- Python 3.12
- Git

### Installation

Clone the repository and enter the project directory:

```bash
git clone <YOUR_REPOSITORY_URL>
cd TinyRAG
```

Create and activate a virtual environment.

**Windows PowerShell:**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project and development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

### Index Documents

The project includes a sample Markdown document in `docs/`.

```powershell
tinyrag index docs
```

### Search Documents

```powershell
tinyrag search "How do I isolate Python dependencies?" --top-k 3
```

### Run Tests

```powershell
python -m pytest
```

### Run Retrieval Evaluation

```powershell
python -m tinyrag.eval_runner
```

## Evaluation

The current evaluation dataset contains five queries over a single sample Markdown document.

Initial results with `k = 1`:

| Metric | Result |
|---|---:|
| Mean Precision@1 | 1.000 |
| Mean Recall@1 | 1.000 |
| Mean Reciprocal Rank | 1.000 |

These results are preliminary and reflect a small evaluation dataset. A larger and more diverse corpus and query set are needed to assess general retrieval quality.

## Current Limitations

- The evaluation corpus is small.
- Search quality has not yet been benchmarked against alternative retrieval methods.
- Answer generation grounded in retrieved context has not yet been implemented.
- The current system is intended as a learning project, not a production-ready RAG service.

## Roadmap

- [x] Markdown document loading and chunking
- [x] Local sentence embeddings
- [x] Vector similarity search
- [x] Persistent index storage
- [x] Retrieval evaluation metrics
- [ ] Expand the document corpus and evaluation dataset
- [ ] Improve retrieval quality and test edge cases
- [ ] Generate answers using retrieved context
- [ ] Include source citations in generated answers
- [ ] Add end-to-end RAG tests

## Project Goals

TinyRAG is an incremental exploration of the building blocks behind retrieval systems. The goal is to understand, implement, evaluate, and document each component rather than treating RAG as a black box.