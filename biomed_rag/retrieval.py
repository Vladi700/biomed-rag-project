"""Hybrid retrieval: dense + BM25 fusion, cross-encoder reranking, merging of adjacent chunks."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from . import config, indexing, llm


@dataclass
class RetrievalConfig:
    k_final: int = 5            # chunks given to the LLM
    k_dense: int = 30           # dense candidates
    k_bm25: int = 30            # BM25 candidates
    k_rerank: int = 30          # fused candidates sent to the cross-encoder
    rrf_k: int = 60             # reciprocal-rank-fusion constant
    min_dense: float = 0.45     # cosine floor on the best dense hit; below -> refuse
    expand_neighbours: bool = True
    use_reranker: bool = True
    reranker_id: str = config.RERANKER_MODEL_ID


def bm25_tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def rrf_fuse(ranked_lists: list[list[tuple[int, float]]], rrf_k: int) -> list[tuple[int, float]]:
    """Reciprocal rank fusion -> [(meta_idx, rrf_score)], best first."""
    fused = {}
    for lst in ranked_lists:
        for rank, (mi, _) in enumerate(lst):
            fused[mi] = fused.get(mi, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(fused.items(), key=lambda x: -x[1])


def group_hits(ranked_ids: list[int], metadata: list[dict], k: int) -> list[dict]:
    """Dedup candidates and merge each into the group whose primary chunk it is adjacent to; stop at `k` groups."""
    seen, groups = set(), []
    for mi in ranked_ids:
        m = metadata[mi]
        key = m["chunk_file"]
        if key in seen:
            continue
        # if an already-selected chunk is adjacent in the same doc, merge
        prev = next((g for g in groups
                     if metadata[g["primary"]]["doc_id"] == m["doc_id"]
                     and abs(metadata[g["primary"]]["chunk_idx"] - m["chunk_idx"]) == 1), None)
        if prev is not None:
            prev["members"].add(mi)
        else:
            groups.append({"primary": mi, "members": {mi}})
        seen.add(key)
        if len(groups) >= k:
            break
    return groups


def expand_neighbours(groups: list[dict], metadata: list[dict], position: dict[tuple[str, int], int]) -> list[dict]:
    """Add the same-document neighbours (chunk_idx +-1) of each primary chunk to its group."""
    for g in groups:
        m = metadata[g["primary"]]
        for step in (-1, 1):
            ni = position.get((m["doc_id"], m["chunk_idx"] + step))
            if ni is not None:
                g["members"].add(ni)
    return groups


def group_text(members, metadata: list[dict], texts: list[str]) -> str:
    """Member texts in document order (by chunk_idx), joined by newlines."""
    ordered = sorted(members, key=lambda mi: metadata[mi]["chunk_idx"])
    return "\n".join(texts[mi] for mi in ordered)


class HybridRetriever:
    """Dense (FAISS) + BM25 retrieval fused with RRF, optionally reranked by a cross-encoder."""

    def __init__(self, embedder, index, metadata: list[dict], texts: list[str],
                 retrieval_config: RetrievalConfig | None = None, reranker=None):
        from rank_bm25 import BM25Okapi

        self.config = retrieval_config or RetrievalConfig()
        self.embedder = embedder
        self.index = index
        self.metadata = metadata
        self.texts = texts
        self.retrieval_texts = [indexing.retrieval_text(text, m) for text, m in zip(texts, metadata)]

        # ---------- Build BM25 index once (aligned with `metadata` order) ----------
        print("Building BM25 index over metadata-enriched chunk texts...")
        self.bm25 = BM25Okapi([bm25_tokenize(t) for t in self.retrieval_texts])
        print(f"BM25 index ready: {len(texts)} chunks")

        # (doc_id, chunk_idx) -> position in metadata (for neighbour expansion)
        self.position = {(m["doc_id"], m["chunk_idx"]): i for i, m in enumerate(metadata)}

        if reranker is None and self.config.use_reranker:
            reranker = CrossEncoder(self.config.reranker_id, device="cuda" if torch.cuda.is_available() else "cpu")
        self.reranker = reranker

    @classmethod
    def from_disk(cls, vector_db_dir: Path, chunk_dir: Path, retrieval_config: RetrievalConfig | None = None,
                  embedding_model_id: str = config.EMBEDDING_MODEL_ID, reranker=None) -> "HybridRetriever":
        """Load the saved index, check it against the chunk files and build the retriever."""
        index, metadata = indexing.load_index(vector_db_dir)
        indexing.validate_index(index, metadata, chunk_dir)
        texts = indexing.load_chunk_texts(metadata, chunk_dir)
        embedder = SentenceTransformer(embedding_model_id)
        return cls(embedder, index, metadata, texts, retrieval_config=retrieval_config, reranker=reranker)

    def dense_candidates(self, query: str, k: int) -> list[tuple[int, float]]:
        """Top-k (meta_idx, cosine) from the FAISS index."""
        q = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, k)
        return [(int(i), float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]

    def bm25_candidates(self, query: str, k: int) -> list[tuple[int, float]]:
        """Top-k (meta_idx, BM25 score)."""
        scores = self.bm25.get_scores(bm25_tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top]

    def search(self, query: str, k: int | None = None) -> list[dict]:
        """Retrieve up to `k` passages for `query`; an empty list means the query was refused."""
        cfg = self.config
        k = cfg.k_final if k is None else k
        dense = self.dense_candidates(query, cfg.k_dense)
        lex = self.bm25_candidates(query, cfg.k_bm25)

        # --- Similarity threshold: refuse weak retrievals ---
        if not dense or dense[0][1] < cfg.min_dense:
            return []

        fused = rrf_fuse([dense, lex], cfg.rrf_k)[:cfg.k_rerank]
        cand_idx = [mi for mi, _ in fused]
        scores = dict(fused)

        # --- Cross-encoder rerank ---
        if self.reranker is not None:
            pairs = [(query, self.retrieval_texts[mi]) for mi in cand_idx]
            rerank_scores = [float(s) for s in self.reranker.predict(pairs)]
            scores = dict(zip(cand_idx, rerank_scores))
            order = np.argsort(rerank_scores)[::-1]
            cand_idx = [cand_idx[i] for i in order]

        # --- Dedup + merge adjacent hits ---
        groups = group_hits(cand_idx, self.metadata, k)

        # --- Neighbour expansion (context before/after each hit) ---
        if cfg.expand_neighbours:
            expand_neighbours(groups, self.metadata, self.position)

        dense_scores = dict(dense)
        hits = []
        for rank, g in enumerate(groups, start=1):
            primary = self.metadata[g["primary"]]
            members = sorted(g["members"], key=lambda mi: self.metadata[mi]["chunk_idx"])
            member_metadata = [self.metadata[mi] for mi in members]
            sections = list(dict.fromkeys(s for m in member_metadata for s in m.get("sections", [])))
            passage_types = list(dict.fromkeys(t for m in member_metadata for t in m.get("passage_types", [])))
            starts = [m["source_offset_start"] for m in member_metadata if m.get("source_offset_start") is not None]
            ends = [m["source_offset_end"] for m in member_metadata if m.get("source_offset_end") is not None]
            hits.append({
                "rank": rank,
                "chunk_file": primary["chunk_file"],
                "doc_id": primary["doc_id"],
                "title": primary.get("title", ""),
                "article_ids": primary.get("article_ids", {}),
                "sections": sections,
                "passage_types": passage_types,
                "source_offset_start": min(starts) if starts else None,
                "source_offset_end": max(ends) if ends else None,
                "chunk_files": [self.metadata[mi]["chunk_file"] for mi in members],
                "score": scores[g["primary"]],
                "dense_score": dense_scores.get(g["primary"]),
                "text": group_text(members, self.metadata, self.texts),
            })
        return hits

    def close(self) -> None:
        """Drop the models and free the memory."""
        self.embedder = None
        self.reranker = None
        llm.free_memory()
