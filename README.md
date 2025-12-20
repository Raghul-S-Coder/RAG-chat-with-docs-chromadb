# RAG-chat-with-docs-chromadb

A Python-based **Retrieval-Augmented Generation (RAG)** project that enables question answering and summarization over user-provided documents using **ChromaDB** and **Hugging Face embeddings**.

---

## Overview

This project loads text documents, converts them into vector embeddings, stores them in **ChromaDB**, and uses a Large Language Model (LLM) to answer user queries or generate summaries based on the retrieved contextual data.

The document loading behavior is configurable through a YAML-based configuration system.

---

## Project Structure

```text
RAG-chat-with-docs-chromadb/
│
├── main.py
│   # Entry point of the application.
│   # Loads documents into ChromaDB (based on config)
│   # and invokes the LLM for summarization or Q&A.
│
├── src/
│   └── vector_db.py
│       # Handles vector database creation, embedding,
│       # and interaction with ChromaDB.
│
├── raw_data/
│   # Contains sample input documents for embedding.
│   # Currently supports .txt files only.
│
├── properties/
│   ├── vector-config.yaml
│   # Configuration file for vector database behavior.
│   #
│   # Example:
│   # load_into_vector: true
│
│   └── vector_config_loader.py
│       # Utility to load and parse vector-config.yaml.
│
└── README.md
```

---

## Configuration

All vector database behavior is controlled via: properties/vector-config.yaml

### Key Property

- **load_into_vector**
  - `true` → Loads documents from `raw_data/` into ChromaDB
  - `false` → Skips document loading and uses existing vectors

This allows flexible control over when embeddings are regenerated.

---

## Embedding Model

The project uses the following Hugging Face embedding model: sentence-transformers/all-MiniLM-L6-v2

This model provides fast and efficient sentence-level embeddings suitable for semantic search and RAG pipelines.

---

## Supported Data Formats

- ✅ `.txt` files
- ❌ Other formats (PDF, DOCX, etc.) are not supported in the current implementation

---

## How It Works

1. Configuration is loaded from `vector-config.yaml`
2. If `load_into_vector` is enabled:
   - Text files from `raw_data/` are embedded
   - Embeddings are stored in ChromaDB
3. User queries are matched against stored vectors
4. Retrieved context is passed to the LLM
5. The LLM generates answers or summaries based on the context

---

## Requirements

- Python 3.8+
- ChromaDB
- Hugging Face `sentence-transformers`
- A supported LLM provider

---

## License

This project is provided for educational and development purposes.
