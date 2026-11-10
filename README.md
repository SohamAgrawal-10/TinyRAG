# TinyRAG

### A lightweight local Retrieval-Augmented Generation (RAG) system for semantic search and question answering over Markdown documents.

TinyRAG is a lightweight, fully local **Retrieval-Augmented Generation (RAG)** pipeline built in Python. It takes Markdown documents, breaks them into meaningful chunks, converts those chunks into semantic embeddings, stores them in a persistent vector index, retrieves the most relevant context for a query, and uses a local language model to generate an answer grounded in the retrieved documents.

The project was built to understand and implement the core components of a RAG system from the ground up rather than relying on an end-to-end framework.

> **Pipeline:** Markdown Documents → Chunking → Embeddings → Vector Index → Semantic Retrieval → Context → Local LLM → Answer + Sources

---

## Table of Contents

- [Project Overview](#project-overview)
- [Why TinyRAG?](#why-tinyrag)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Index Documents](#1-index-documents)
  - [Search Documents](#2-search-documents)
  - [Ask Questions](#3-ask-questions)
- [Evaluation](#evaluation)
- [Example](#example)
- [Design Decisions](#design-decisions)
- [Challenges](#challenges)
- [Testing](#testing)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Credits](#credits)
- [License](#license)

---

## Project Overview

Large Language Models can generate useful answers, but they do not automatically have access to private or domain-specific documents.

A RAG system addresses this by retrieving relevant information from an external knowledge source and providing that information to the language model as context before generating an answer.

TinyRAG implements this process locally:

```text
                ┌────────────────────┐
                │   Markdown Docs    │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Document Chunking  │
                │ Heading-aware      │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Semantic Embedding │
                │ all-MiniLM-L6-v2   │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Persistent Vector  │
                │      Index         │
                └─────────┬──────────┘
                          │
                    User Question
                          │
                          ▼
                ┌────────────────────┐
                │ Semantic Retrieval │
                └─────────┬──────────┘
                          │
                    Relevant Context
                          │
                          ▼
                ┌────────────────────┐
                │     Local LLM      │
                │      FLAN-T5       │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Answer + Sources   │
                └────────────────────┘
```

---

## Why TinyRAG?

The goal was not simply to build another application using a RAG framework.

TinyRAG was designed as a learning-focused implementation of the underlying retrieval pipeline:

- How documents are converted into retrievable chunks
- How semantic embeddings represent text numerically
- How vector similarity can be used for retrieval
- How retrieved context is passed to a language model
- How relevance thresholds can prevent unrelated questions from reaching the generator
- How retrieval quality can be evaluated independently from generation

This makes the project useful for understanding the components that sit underneath larger RAG frameworks.

---

## Features

### Document Processing

- Markdown document ingestion
- Heading-aware document chunking
- Source and line-range metadata preservation
- Support for multiple Markdown files

### Semantic Retrieval

- Sentence-transformer embeddings
- `all-MiniLM-L6-v2` embedding model
- 384-dimensional embedding vectors
- Persistent local vector index
- Cosine-similarity-based retrieval
- Configurable `top-k` retrieval
- Minimum similarity threshold

### RAG Generation

- Local language model inference
- Retrieved chunks supplied as generation context
- Answers constrained to the provided document context
- Source attribution for retrieved passages

### Evaluation

- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- In-scope query evaluation
- Out-of-scope query rejection
- Configurable relevance threshold

### Developer Experience

- Command-line interface
- Persistent indexes
- Automated test suite
- Lightweight Python architecture
- No external API key required for the core pipeline

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Embeddings | Sentence Transformers |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Search | NumPy-based vector index |
| Generation | Hugging Face Transformers |
| Generation Model | `google/flan-t5-base` |
| CLI | Typer |
| Terminal UI | Rich |
| Testing | Pytest |
| Code Quality | Ruff |
| Version Control | Git + GitHub |

---

## Project Structure

```text
TinyRAG/
│
├── docs/
│   ├── python_basics.md
│   └── git_basics.md
│
├── data/
│   └── index/
│       ├── vectors.npy
│       └── metadata.json
│
├── src/
│   └── tinyrag/
│       ├── __init__.py
│       ├── chunking.py
│       ├── embeddings.py
│       ├── index.py
│       ├── storage.py
│       ├── search.py
│       ├── generator.py
│       ├── evaluation.py
│       ├── eval_runner.py
│       └── cli.py
│
├── tests/
│   ├── fixtures/
│   │   ├── queries.json
│   │   └── out_of_scope_queries.json
│   ├── test_chunking.py
│   ├── test_cli.py
│   ├── test_embeddings.py
│   ├── test_evaluation.py
│   ├── test_index.py
│   └── test_storage.py
│
├── pyproject.toml
└── README.md
```

---

## Installation

### Requirements

- Python `3.12`
- Git
- A machine capable of running the embedding and generation models locally

TinyRAG does not require an external API key for its core functionality.

### 1. Clone the repository

```bash
git clone https://github.com/SohamAgrawal-10/TinyRAG.git
cd TinyRAG
```

### 2. Create a virtual environment

#### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the project

```bash
pip install -e .
```

### 4. Install development dependencies

```bash
pip install -e ".[dev]"
```

---

## Usage

TinyRAG provides three primary CLI commands:

```text
tinyrag index
tinyrag search
tinyrag ask
```

---

### 1. Index Documents

Place Markdown files inside the `docs/` directory and build the vector index:

```bash
tinyrag index
```

Example output:

```text
Embedding 20 chunks...
Indexed 20 chunks.
Saved index to data/index
```

The documents are:

1. Loaded
2. Split into meaningful chunks
3. Converted into embedding vectors
4. Stored in the persistent local index

The index can then be reused without re-processing the documents for every query.

---

### 2. Search Documents

Use semantic search to retrieve relevant passages:

```bash
tinyrag search "How do I create a new branch in Git?"
```

You can control the number of retrieved results:

```bash
tinyrag search "How do I create a new branch in Git?" --top-k 3
```

You can also apply a minimum similarity threshold:

```bash
tinyrag search "How do I create a new branch in Git?" --top-k 3 --min-score 0.45
```

Example:

```text
TinyRAG Search Results

Rank  Score   Source
1     0.6671  git_basics.md:13-22

Passage:
## Branches

A branch provides an independent line of development...
Use `git branch` to inspect existing branches.
Use `git switch -c <branch-name>` to create a new branch...
```

---

### 3. Ask Questions

The `ask` command combines retrieval and generation into a complete RAG pipeline:

```bash
tinyrag ask "How do I create a new branch in Git?"
```

TinyRAG first retrieves the most relevant document context and then passes that context to the local language model.

Example:

```text
Generating answer...

Answer
git switch -c branch-name

Sources: git_basics.md:13-22
```

The important distinction is that the language model is **not asked to answer from its general knowledge alone**. It receives the retrieved document context as part of the generation prompt.

---

## Evaluation

TinyRAG includes a small evaluation suite for measuring retrieval quality.

The evaluation set contains:

- **13 in-scope queries** with known relevant documents
- **8 out-of-scope queries** that should be rejected

The current evaluation uses a minimum relevance threshold of `0.45`.

### Results

| Metric | Result |
|---|---:|
| In-scope queries | 13 |
| Precision@1 | **1.000** |
| Recall@1 | **1.000** |
| MRR | **1.000** |
| Out-of-scope queries | 8 |
| Out-of-scope rejection rate | **1.000** |

All 13 in-scope queries retrieved the expected relevant passage at rank 1, while all 8 out-of-scope queries were rejected using the configured similarity threshold.

> **Note:** These results come from a small, manually curated evaluation set and should not be interpreted as a general benchmark of retrieval performance.

---

## Example RAG Flow

For a query such as:

```text
How do I create a new branch in Git?
```

TinyRAG performs the following operations:

### Step 1 — Embed the query

The question is converted into a 384-dimensional semantic vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

### Step 2 — Retrieve relevant chunks

The query vector is compared against the stored document vectors using cosine similarity.

The most relevant chunk is:

```text
git_basics.md:13-22
```

### Step 3 — Apply relevance filtering

If the similarity score is below the configured threshold, TinyRAG does not generate an answer.

This prevents obviously unrelated questions from being passed to the language model.

### Step 4 — Construct the RAG prompt

The retrieved document content is combined with the user's question:

```text
Context:
[git_basics.md:13-22]
...

Question:
How do I create a new branch in Git?

Answer:
```

### Step 5 — Generate the answer

The local FLAN-T5 model generates an answer based on the supplied context.

### Step 6 — Return the source

TinyRAG returns the generated answer together with the source document and line range.

---

## Design Decisions

### Why Markdown?

Markdown provides a simple, human-readable knowledge source while still containing useful structural information through headings.

This makes it a good format for experimenting with document chunking without introducing unnecessary complexity.

### Why heading-aware chunking?

Naively splitting documents into fixed-size chunks can separate related information.

Heading-aware chunking attempts to preserve semantic boundaries, allowing sections such as:

```text
## Branches
## Commits
## Merge Conflicts
```

to remain meaningful retrieval units.

### Why `all-MiniLM-L6-v2`?

The model provides a useful balance between:

- Embedding quality
- Model size
- Inference speed
- Local hardware requirements

This makes it appropriate for a lightweight local project.

### Why local generation?

The project was designed to demonstrate that a RAG pipeline can be run without depending on a paid external LLM API.

The retrieval and generation stages therefore run locally using open-source models.

### Why similarity thresholds?

Retrieval systems will always return the "closest" result, even when a query has nothing to do with the indexed documents.

A minimum similarity threshold allows TinyRAG to distinguish between:

```text
Relevant question
        ↓
Retrieve context
        ↓
Generate answer
```

and:

```text
Unrelated question
        ↓
No sufficiently relevant context
        ↓
Reject query
```

---

## Challenges

### 1. Retrieval vs. Generation

One of the key challenges was distinguishing between a retrieval problem and a generation problem.

A language model cannot reliably answer a question if the retrieved chunk does not contain the necessary information.

For example, initially the Git documentation explained that `git switch` changes branches but did not explicitly explain how to create one. The retriever correctly found the branch section, but the generator could not produce a reliable answer from incomplete context.

This highlighted an important RAG principle:

> **Good generation cannot compensate for bad or incomplete retrieval context.**

### 2. Out-of-scope queries

A semantic search engine will generally return its closest match even for unrelated questions.

A similarity threshold was therefore introduced to reject queries that do not have sufficiently relevant context.

### 3. Keeping the system lightweight

The project intentionally avoids large infrastructure such as hosted vector databases, external APIs, or distributed services.

This keeps the system easy to understand, run, and modify while exposing the fundamental components of a RAG pipeline.

---

## Testing

TinyRAG uses `pytest` for automated testing.

The current test suite contains **21 tests**, covering:

- Document chunking
- Embedding functionality
- Vector indexing
- Persistent storage
- Retrieval evaluation
- CLI behavior

Run the test suite with:

```bash
python -m pytest
```

Current result:

```text
21 passed
```

---

## Limitations

TinyRAG is intentionally lightweight and has several limitations.

### Retrieval

The current vector index is designed for a small local document collection and is not intended to replace production vector databases.

### Evaluation

The current evaluation dataset is relatively small and manually curated.

Larger datasets and domain-diverse queries would be required for more rigorous benchmarking.

### Generation

The local FLAN-T5 model is relatively small compared with modern production-grade instruction-tuned LLMs. Consequently, generated answers can occasionally be incomplete or awkwardly phrased even when the correct context is retrieved.

### Document formats

The current ingestion pipeline focuses primarily on Markdown documents.

---

## Future Improvements

Potential extensions include:

- Support for PDFs and additional document formats
- Improved chunking strategies
- Hybrid keyword + semantic retrieval
- Re-ranking retrieved passages
- Larger and more diverse evaluation datasets
- Better local instruction-tuned generation models
- Streaming generation
- Conversation-aware RAG
- Retrieval and generation latency benchmarking
- A lightweight web interface
- Support for larger-scale vector databases

---

## Credits

### Author

**Soham Agrawal**


- GitHub: [SohamAgrawal-10](https://github.com/SohamAgrawal-10)
- Project: [TinyRAG](https://github.com/SohamAgrawal-10/TinyRAG)

### Technologies & References

TinyRAG builds upon the following open-source projects and models:

- [Sentence Transformers](https://www.sbert.net/) — semantic text embeddings
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) — local language model inference
- [FLAN-T5](https://huggingface.co/google/flan-t5-base) — local text-to-text generation
- [Typer](https://typer.tiangolo.com/) — command-line interface
- [Rich](https://rich.readthedocs.io/) — terminal formatting
- [Pytest](https://docs.pytest.org/) — automated testing

---

## License

This project is intended as an educational and portfolio project.