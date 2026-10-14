# Biomedical RAG Project

A retrieval-augmented generation pipeline for answering questions over recent open-access PubMed Central articles about machine learning in biomedicine.

The project uses:

- PubMedBERT embeddings and FAISS for dense retrieval
- BM25 for lexical retrieval
- reciprocal rank fusion and `BAAI/bge-reranker-base` for reranking
- `LiquidAI/LFM2-1.2B-RAG` for grounded answer generation
- `Qwen/Qwen3-4B-Instruct-2507` for synthetic test-set generation and evaluation

## Structure

- `notebook_latest (2).ipynb` — main Jupyter/Colab presentation and execution workflow
- `biomed_rag/` — reusable corpus, chunking, indexing, retrieval, generation, and evaluation modules

Downloaded PMC articles, generated chunks, vector indexes, model outputs, archives, and local credentials are intentionally excluded from version control.

## Running in Google Colab

Set `NCBI_EMAIL` to a contact address before downloading articles from PMC:

```python
import os
os.environ["NCBI_EMAIL"] = "your-email@example.com"
```

1. Zip the `biomed_rag/` directory as `biomed_rag.zip`.
2. Upload it with the required corpus archives described in the notebook.
3. Open `notebook_latest (2).ipynb` in Colab and run the required cells in order.

The notebook installs its Python dependencies and documents the design choices, retrieval configuration, prompts, and evaluation procedure.
