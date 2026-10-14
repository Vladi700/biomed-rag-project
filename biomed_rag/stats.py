"""Corpus and chunk statistics, with validation checks on the chunks."""

import re
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from . import config


def describe(values) -> dict:
    """Count, mean, median, std, min and max as plain Python numbers."""
    values = np.asarray(values)
    if values.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0,
            "max": 0,
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "min": int(values.min()),
        "max": int(values.max()),
    }


def print_summary(label: str, stats: dict) -> None:
    """Print one `describe` result on a single line."""
    print(
        f"{label:36s} "
        f"n={stats['count']:5d}  "
        f"mean={stats['mean']:9.1f}  "
        f"median={stats['median']:8.1f}  "
        f"std={stats['std']:8.1f}  "
        f"min={stats['min']:6d}  "
        f"max={stats['max']:6d}"
    )


def chunk_doc_id(path: Path) -> str:
    return path.stem.rsplit("_chunk_", 1)[0]


def corpus_statistics(chunk_dir: Path, raw_dir: Path, tokenizer_id: str = config.EMBEDDING_MODEL_ID,
                      max_chunk_tokens: int = 512, model_limit: int = 512) -> dict:
    """Size statistics of chunks and raw articles, plus validation of the chunks."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.model_max_length = int(1e9)

    chunk_files = sorted(Path(chunk_dir).glob("PMC*_chunk_*.txt"))
    raw_files = sorted(Path(raw_dir).glob("PMC*.txt"))

    chunk_chars = []
    chunk_words = []
    chunk_content_tokens = []
    chunk_model_tokens = []
    chunks_per_doc = Counter()

    oversized_content_chunks = []
    oversized_model_inputs = []
    mixed_document_chunks = []

    for path in chunk_files:
        text = path.read_text(encoding="utf-8")
        doc_id = chunk_doc_id(path)

        content_tokens = len(
            tokenizer.encode(text, add_special_tokens=False, truncation=False)
        )
        model_tokens = len(
            tokenizer.encode(text, add_special_tokens=True, truncation=False)
        )

        chunk_chars.append(len(text))
        chunk_words.append(len(text.split()))
        chunk_content_tokens.append(content_tokens)
        chunk_model_tokens.append(model_tokens)
        chunks_per_doc[doc_id] += 1

        if content_tokens > max_chunk_tokens:
            oversized_content_chunks.append({"chunk_file": path.name, "tokens": content_tokens})
        if model_tokens > model_limit:
            oversized_model_inputs.append({"chunk_file": path.name, "tokens": model_tokens})

        doc_ids_in_name = re.findall(r"PMC\d+", doc_id)
        if len(set(doc_ids_in_name)) > 1:
            mixed_document_chunks.append(path.name)

    raw_chars = []
    raw_words = []
    raw_tokens = []

    for path in raw_files:
        text = path.read_text(encoding="utf-8")
        raw_chars.append(len(text))
        raw_words.append(len(text.split()))
        raw_tokens.append(
            len(tokenizer.encode(text, add_special_tokens=False, truncation=False))
        )

    chunk_doc_ids = set(chunks_per_doc)
    raw_doc_ids = {path.stem for path in raw_files}

    return {
        "n_documents": len(raw_files),
        "n_documents_with_chunks": len(chunk_doc_ids),
        "n_chunks": len(chunk_files),
        "average_chunks_per_document": len(chunk_files) / len(chunk_doc_ids) if chunk_doc_ids else None,
        "chunk_statistics": {
            "characters": describe(chunk_chars),
            "words": describe(chunk_words),
            "content_tokens": describe(chunk_content_tokens),
            "tokens_with_special_tokens": describe(chunk_model_tokens),
        },
        "document_statistics": {
            "chunks": describe(list(chunks_per_doc.values())),
            "characters": describe(raw_chars),
            "words": describe(raw_words),
            "tokens": describe(raw_tokens),
        },
        "validation": {
            "mixed_document_chunks": mixed_document_chunks,
            "chunks_over_max_tokens": oversized_content_chunks,
            "inputs_over_model_limit": oversized_model_inputs,
            "documents_without_chunks": sorted(raw_doc_ids - chunk_doc_ids),
            "unknown_chunk_documents": sorted(chunk_doc_ids - raw_doc_ids),
        },
    }


def print_corpus_report(stats: dict, max_chunk_tokens: int = 512, model_limit: int = 512) -> None:
    """Print the result of `corpus_statistics`."""
    chunk_stats = stats["chunk_statistics"]
    doc_stats = stats["document_statistics"]
    validation = stats["validation"]

    print("=== CORPUS OVERVIEW ===")
    print(f"Raw documents:                   {stats['n_documents']}")
    print(f"Documents represented in chunks:{stats['n_documents_with_chunks']:5d}")
    print(f"Total chunks:                    {stats['n_chunks']}")
    average = stats["average_chunks_per_document"]
    print(f"Average chunks per document:     {average:.2f}" if average is not None
          else "Average chunks per document:     N/A")

    print("\n=== CHUNK-LEVEL STATISTICS ===")
    print_summary("Characters per chunk", chunk_stats["characters"])
    print_summary("Words per chunk", chunk_stats["words"])
    print_summary("Content tokens per chunk", chunk_stats["content_tokens"])
    print_summary("Tokens with special tokens", chunk_stats["tokens_with_special_tokens"])

    print("\n=== DOCUMENT-LEVEL STATISTICS ===")
    print_summary("Chunks per document", doc_stats["chunks"])
    print_summary("Characters per raw document", doc_stats["characters"])
    print_summary("Words per raw document", doc_stats["words"])
    print_summary("Tokens per raw document", doc_stats["tokens"])

    print("\n=== VALIDATION ===")
    print(f"Mixed-document chunks:           {len(validation['mixed_document_chunks'])}")
    print(f"Chunks over {max_chunk_tokens} content tokens:  {len(validation['chunks_over_max_tokens'])}")
    print(
        f"Inputs over model limit "
        f"({model_limit}):     {len(validation['inputs_over_model_limit'])}"
    )
    print(f"Documents without chunks:        {len(validation['documents_without_chunks'])}")
    print(f"Chunks with unknown document IDs:{len(validation['unknown_chunk_documents']):5d}")

    if validation["mixed_document_chunks"]:
        print("\nMixed-document examples:")
        for name in validation["mixed_document_chunks"][:10]:
            print(f"  {name}")

    if validation["chunks_over_max_tokens"]:
        print(f"\nChunks over {max_chunk_tokens} content tokens:")
        for item in validation["chunks_over_max_tokens"][:10]:
            print(f"  {item['chunk_file']}: {item['tokens']} tokens")

    if validation["inputs_over_model_limit"]:
        print("\nChunks truncated by the embedding model:")
        for item in validation["inputs_over_model_limit"][:10]:
            print(f"  {item['chunk_file']}: {item['tokens']} tokens")
