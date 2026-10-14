"""Paragraph-aware chunking of article texts into token-bounded segments."""

import bisect
from pathlib import Path

from transformers import AutoTokenizer

from . import config
from .utils import load_json, save_json


def paragraph_end_tokens(text: str, offsets: list[tuple[int, int]]) -> list[int]:
    """Token indices (exclusive) at which each '\n\n'-separated paragraph ends."""
    char_ends = [e for _, e in offsets]
    ends, pos = [], 0
    for para in text.split("\n\n"):
        pos += len(para)
        c = bisect.bisect_right(char_ends, pos)  # last token whose char end is <= this paragraph's end
        if c:
            ends.append(c)
        pos += 2  # skip the '\n\n'
    return sorted(set(ends))


def chunk_spans(n: int, boundaries: list[int], word_ids: list[int | None], chunk_tokens: int = 512,
                overlap_tokens: int = 64, min_chunk_tokens: int = 128) -> list[tuple[int, int]]:
    """Token spans (start, end) of at most `chunk_tokens`, ending on paragraph boundaries where possible."""
    cuts = sorted(set(boundaries) | {n})
    spans, start, prev_end = [], 0, 0
    while True:
        limit = min(start + chunk_tokens, n)
        end = max((c for c in cuts if prev_end < c <= limit and c - start >= min_chunk_tokens), default=limit)
        if end < n and end == limit:                       # hard cut: don't split a word
            e = end
            while e > prev_end + 1 and word_ids[e] is not None and word_ids[e] == word_ids[e - 1]:
                e -= 1
            if e - start >= min_chunk_tokens and e > prev_end:
                end = e
        spans.append((start, end))
        if end >= n:
            break
        prev_end = end
        start = max(start + 1, end - overlap_tokens)
        while start < prev_end and word_ids[start] is not None and word_ids[start] == word_ids[start - 1]:
            start += 1                                       # overlap starts on a word boundary
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < min_chunk_tokens:
        spans[-1] = (max(0, n - chunk_tokens), n)            # re-anchor short tail to a full window
    return spans


def chunk_document_records(text: str, tokenizer, chunk_tokens: int = 512, overlap_tokens: int = 64,
                           min_chunk_tokens: int = 128) -> list[dict]:
    """Split an article into original-text slices with their flattened character spans."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    n = len(offsets)
    if n == 0:
        return []

    boundaries = paragraph_end_tokens(text, offsets)
    spans = chunk_spans(n, boundaries, enc.word_ids(), chunk_tokens, overlap_tokens, min_chunk_tokens)
    return [
        {
            "text": text[offsets[a][0]:offsets[b - 1][1]],
            "flat_char_start": offsets[a][0],
            "flat_char_end": offsets[b - 1][1],
        }
        for a, b in spans
    ]


def chunk_document(text: str, tokenizer, chunk_tokens: int = 512, overlap_tokens: int = 64,
                   min_chunk_tokens: int = 128) -> list[str]:
    """Split one article into chunk texts (slices of the original text, never decoded)."""
    return [
        record["text"]
        for record in chunk_document_records(text, tokenizer, chunk_tokens, overlap_tokens, min_chunk_tokens)
    ]


def _chunk_metadata(chunk: dict, article: dict) -> dict:
    overlapping = [
        passage for passage in article.get("passages", [])
        if passage["flat_char_start"] < chunk["flat_char_end"]
        and passage["flat_char_end"] > chunk["flat_char_start"]
    ]
    sections = list(dict.fromkeys(p["section_type"] for p in overlapping if p["section_type"]))
    passage_types = list(dict.fromkeys(p["passage_type"] for p in overlapping if p["passage_type"]))
    source_passages = []
    for passage in overlapping:
        overlap_start = max(chunk["flat_char_start"], passage["flat_char_start"])
        overlap_end = min(chunk["flat_char_end"], passage["flat_char_end"])
        source_passages.append({
            "passage_idx": passage["passage_idx"],
            "section_type": passage["section_type"],
            "passage_type": passage["passage_type"],
            "passage_offset_start": passage["source_offset_start"],
            "passage_offset_end": passage["source_offset_end"],
            "chunk_offset_start": passage["source_offset_start"] + overlap_start - passage["flat_char_start"],
            "chunk_offset_end": passage["source_offset_start"] + overlap_end - passage["flat_char_start"],
        })
    return {
        "title": article.get("title", ""),
        "article_ids": article.get("article_ids", {}),
        "sections": sections,
        "passage_types": passage_types,
        "flat_char_start": chunk["flat_char_start"],
        "flat_char_end": chunk["flat_char_end"],
        "source_offset_start": min((p["chunk_offset_start"] for p in source_passages), default=None),
        "source_offset_end": max((p["chunk_offset_end"] for p in source_passages), default=None),
        "source_passages": source_passages,
    }


def chunk_corpus(src_dir: Path, dst_dir: Path, tokenizer_id: str = config.EMBEDDING_MODEL_ID,
                 chunk_tokens: int = 512, overlap_tokens: int = 64, min_chunk_tokens: int = 128) -> int:
    """Chunk every article in `src_dir` into `dst_dir/{doc_id}_chunk_{i:04d}.txt`; return the chunk count."""
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    doc_paths = sorted(src_dir.glob("*.txt"))
    if not doc_paths:
        raise FileNotFoundError(f"No *.txt files in {src_dir}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    assert tokenizer.is_fast, "offset mapping requires a fast tokenizer"
    tokenizer.model_max_length = int(1e9)

    dst_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*_chunk_*.txt", "*_chunk_*.metadata.json"):
        for old in dst_dir.glob(pattern):
            old.unlink()

    total_chunks = 0
    for doc_path in doc_paths:
        doc_id = doc_path.stem
        text = doc_path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        metadata_path = src_dir / f"{doc_id}.metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing structural metadata for {doc_path.name}; rebuild the texts stage")
        article = load_json(metadata_path)

        chunks = chunk_document_records(text, tokenizer, chunk_tokens, overlap_tokens, min_chunk_tokens)
        for i, chunk in enumerate(chunks):
            chunk_file = f"{doc_id}_chunk_{i:04d}.txt"
            (dst_dir / chunk_file).write_text(chunk["text"], encoding="utf-8")
            metadata = {
                "chunk_file": chunk_file,
                "doc_id": doc_id,
                "chunk_idx": i,
                **_chunk_metadata(chunk, article),
            }
            save_json(metadata, dst_dir / f"{Path(chunk_file).stem}.metadata.json")
            total_chunks += 1

    print(f"Generated {total_chunks} chunks in {dst_dir}.")
    return total_chunks
