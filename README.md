# TinyRAG

> **A lightweight local Retrieval-Augmented Generation (RAG) system for semantic search and question answering over Markdown documents.**

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg)
![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-Embeddings-orange.svg)
![Pytest](https://img.shields.io/badge/Tests-21%20passed-success.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

---

## 1. Project Overview

TinyRAG is a lightweight, fully local **Retrieval-Augmented Generation pipeline** built in Python.

It takes Markdown documents, converts them into semantically searchable chunks, retrieves relevant context for a question, and uses a local language model to generate an answer grounded in the retrieved documents.

The complete pipeline is:

```text
Markdown Documents
        |
        v
Document Chunking
        |
        v
Semantic Embeddings
        |
        v
Persistent Vector Index
        |
        v
Semantic Retrieval
        |
        v
Relevant Context
        |
        v
Local LLM
        |
        v
Answer + Sources
```

---

## 2. Why I Built This

Rather than using an end-to-end RAG framework, I wanted to understand what actually happens inside a retrieval-augmented system.

The project focuses on the core pipeline:

**Document Processing → Embeddings → Vector Search → Retrieval → Context Construction → Generation → Evaluation**

This makes TinyRAG a small but complete implementation of the fundamental ideas behind modern RAG systems.

---

## 3. Project Architecture

```text
                    Markdown Documents
                           |
                           v
                  Heading-Aware Chunking
                           |
                           v
                  Sentence Embeddings
                all-MiniLM-L6-v2 (384D)
                           |
                           v
                  Persistent Vector Index
                           |
                    User Question
                           |
                           v
                    Query Embedding
                           |
                           v
                   Cosine Similarity
                           |
                           v
                  Relevant Context Chunks
                           |
                           v
                     Relevance Filter
                           |
                           v
                      Local FLAN-T5
                           |
                           v
                    Answer + Sources
```

---

## 4. Key Features

### Document Processing

- Markdown document ingestion
- Heading-aware chunking
- Source and line-range metadata
- Multiple document support

### Semantic Retrieval

- 384-dimensional MiniLM embeddings
- Persistent local vector index
- Cosine similarity search
- Configurable `top-k` retrieval
- Minimum similarity threshold

### RAG Generation

- Local Hugging Face model inference
- Context-grounded answer generation
- Source attribution
- Rejection of insufficiently relevant context

### Evaluation

- Precision@K
- Recall@K
- Mean Reciprocal Rank
- In-scope query evaluation
- Out-of-scope query rejection

---

## 5. RAG Pipeline

### 5.1 Document Chunking

Markdown files are split into semantically meaningful chunks based on their heading structure.

Each chunk retains metadata such as:

```text
Source file
Start line
End line
Text
```

This allows retrieved passages to be traced back to their original location.

### 5.2 Semantic Embeddings

Each chunk is converted into a 384-dimensional vector using:

`sentence-transformers/all-MiniLM-L6-v2`

The same embedding model is used to encode user queries.

### 5.3 Vector Retrieval

TinyRAG compares the query embedding against stored document embeddings using cosine similarity.

The highest-scoring chunks become the context supplied to the generator.

### 5.4 Relevance Filtering

Semantic search will always return the closest match, even when the question is unrelated to the documents.

TinyRAG therefore applies a minimum similarity threshold before generation.

```text
Relevant query
      |
      v
Retrieve context
      |
      v
Generate answer


Unrelated query
      |
      v
Low similarity
      |
      v
Reject
```

### 5.5 Local Generation

Retrieved passages are combined with the user's question and passed to a local FLAN-T5 model.

The generated response is returned together with the retrieved source locations.

---

## 6. Evaluation

TinyRAG includes a small curated retrieval benchmark containing:

- **13 in-scope queries**
- **8 out-of-scope queries**

Current results:

| Metric | Result |
|---|---:|
| Precision@1 | **1.000** |
| Recall@1 | **1.000** |
| MRR | **1.000** |
| Out-of-scope rejection | **100%** |
| Automated tests | **21/21** |

> These results are based on a small manually curated evaluation set and are intended to validate the current retrieval pipeline rather than serve as a general benchmark.

---

## 7. Example

Query:

```text
How do I create a new branch in Git?
```

TinyRAG retrieves:

```text
git_basics.md:13-22
```

and generates:

```text
git switch -c branch-name
```

with the retrieved source attached to the answer.

---

## 8. Technology Stack

### Programming

- Python 3.12

### Machine Learning

- Sentence Transformers
- Hugging Face Transformers
- FLAN-T5
- NumPy

### Retrieval

- Vector embeddings
- Cosine similarity
- Persistent local vector index

### Development

- Typer
- Rich
- Pytest
- Ruff
- Git
- GitHub

---

## 9. Project Structure

```text
tinyrag/
|
├── docs/
│   ├── python_basics.md
│   └── git_basics.md
|
├── src/
│   └── tinyrag/
│       ├── chunking.py
│       ├── embeddings.py
│       ├── index.py
│       ├── storage.py
│       ├── search.py
│       ├── generator.py
│       ├── evaluation.py
│       ├── eval_runner.py
│       └── cli.py
|
├── tests/
│   ├── fixtures/
│   ├── test_chunking.py
│   ├── test_cli.py
│   ├── test_embeddings.py
│   ├── test_evaluation.py
│   ├── test_index.py
│   └── test_storage.py
|
├── data/
│   └── index/
|
├── pyproject.toml
└── README.md
```

---

## 10. Installation

### Requirements

- Python 3.12+
- Git
- Internet connection for downloading the Hugging Face models

### Clone the repository

```bash
git clone https://github.com/SohamAgrawal-10/TinyRAG.git
cd TinyRAG
```

### Create a virtual environment

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

### Install dependencies

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

---

## 11. Usage

TinyRAG provides three primary CLI commands:

```text
tinyrag index
tinyrag search
tinyrag ask
```

### 11.1 Build the Index

Place Markdown files inside the `docs/` directory and run:

```bash
tinyrag index
```

This will:

1. Load the Markdown documents
2. Split them into semantic chunks
3. Generate embeddings
4. Build the vector index
5. Persist the index locally

### 11.2 Semantic Search

Search the indexed documents:

```bash
tinyrag search "How do I create a new branch in Git?"
```

Specify the number of results:

```bash
tinyrag search "How do I create a new branch in Git?" --top-k 3
```

Apply a relevance threshold:

```bash
tinyrag search "How do I create a new branch in Git?" --top-k 3 --min-score 0.45
```

### 11.3 Ask a Question

The `ask` command combines retrieval and generation into a complete RAG pipeline:

```bash
tinyrag ask "How do I create a new branch in Git?"
```

Example:

```text
Generating answer...

Answer
git switch -c branch-name

Sources: git_basics.md:13-22
```

### 11.4 Run Tests

Run the automated test suite:

```bash
python -m pytest
```

Current result:

```text
21 passed
```

---

## 12. Challenges & Design Decisions

### Retrieval vs. Generation

One of the main lessons from the project was that generation quality depends heavily on retrieval quality.

Initially, the Git documentation did not explicitly contain the command for creating a new branch. The retriever correctly found the relevant section, but the generator could not reliably answer the question because the required information was missing from the retrieved context.

This reinforced a central RAG principle:

> **A language model cannot reliably generate an answer from context that does not contain the answer.**

### Local-First Design

The project intentionally avoids paid APIs and hosted infrastructure.

Embeddings, retrieval, and generation are performed locally, making the system easier to experiment with and understand.

### Relevance Thresholding

A semantic search engine can return a result even for an unrelated question.

The relevance threshold provides a simple mechanism for preventing obviously unrelated queries from reaching the generator.

---

## 13. Limitations

TinyRAG is intentionally lightweight and is not intended to be a production-scale RAG platform.

Current limitations include:

- Small local vector index
- Markdown-focused ingestion
- Small evaluation dataset
- Basic cosine-similarity retrieval
- No re-ranking stage
- Relatively small local generation model
- No conversation memory
- No hybrid keyword + semantic retrieval

---

## 14. Future Improvements

- [ ] PDF and additional document support
- [ ] Hybrid keyword + semantic retrieval
- [ ] Retrieval re-ranking
- [ ] Larger evaluation benchmark
- [ ] Better local instruction-tuned model
- [ ] Streaming generation
- [ ] Conversational RAG
- [ ] Retrieval latency benchmarking
- [ ] Lightweight web interface
- [ ] Larger-scale vector database support

---

## 15. Credits & References

TinyRAG uses the following open-source technologies:

- [Sentence Transformers](https://www.sbert.net/) — semantic embeddings
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) — local model inference
- [FLAN-T5](https://huggingface.co/google/flan-t5-base) — text generation
- [Typer](https://typer.tiangolo.com/) — CLI
- [Rich](https://rich.readthedocs.io/) — terminal interface
- [Pytest](https://docs.pytest.org/) — testing

---

## 16. Author

**Soham Agrawal**

Computer Science, BITS Pilani

GitHub: [@SohamAgrawal-10](https://github.com/SohamAgrawal-10)

---

## 17. License

This project is primarily an educational and portfolio project.
