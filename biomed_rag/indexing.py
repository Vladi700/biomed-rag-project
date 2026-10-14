"""Dense FAISS index over the chunks, with JSON chunk metadata."""

from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from . import config
from .utils import StaleArtifactError, load_json, save_json


def chunk_metadata(paths: list[Path]) -> list[dict]:
    """Load the structural metadata sidecar for each chunk path."""
    metadata = []
    for path in paths:
        sidecar = path.with_name(f"{path.stem}.metadata.json")
        if not sidecar.exists():
            raise FileNotFoundError(f"Missing metadata for {path.name}; rebuild the chunks stage")
        record = load_json(sidecar)
        if record.get("chunk_file") != path.name:
            raise StaleArtifactError(f"{sidecar} does not describe {path.name}; rebuild the chunks stage")
        metadata.append(record)
    return metadata


def retrieval_text(text: str, metadata: dict) -> str:
    """Clean prose prefixed by semantic document structure for retrieval models."""
    prefix = []
    if metadata.get("title"):
        prefix.append(f"Article: {metadata['title']}")
    if metadata.get("sections"):
        prefix.append(f"Section: {', '.join(metadata['sections'])}")
    if metadata.get("passage_types"):
        prefix.append(f"Passage type: {', '.join(metadata['passage_types'])}")
    return "\n".join([*prefix, text])


def build_index(chunk_dir: Path, vector_db_dir: Path, model_id: str = config.EMBEDDING_MODEL_ID,
                batch_size: int = 128) -> faiss.Index:
    """Embed all chunks and save the FAISS index and the chunk metadata into `vector_db_dir`."""
    model = SentenceTransformer(model_id)

    chunk_paths = sorted(Path(chunk_dir).glob("*_chunk_*.txt"))
    metadata = chunk_metadata(chunk_paths)
    texts = [
        retrieval_text(path.read_text(encoding="utf-8"), record)
        for path, record in zip(chunk_paths, metadata)
    ]
    print(f"Embedding {len(texts)} chunks with title and section metadata...")

    # normalize_embeddings=True -> inner product == cosine similarity
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    dim = embeddings.shape[1]                     # 768 for this model
    index = faiss.IndexFlatIP(dim)                # exact cosine search
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors, dim {dim}")

    vector_db_dir = Path(vector_db_dir)
    vector_db_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(vector_db_dir / config.INDEX_PATH.name))
    save_json(metadata, vector_db_dir / config.METADATA_PATH.name)
    return index


def load_index(vector_db_dir: Path) -> tuple[faiss.Index, list[dict]]:
    """Load the FAISS index and the chunk metadata saved by `build_index`."""
    vector_db_dir = Path(vector_db_dir)
    index = faiss.read_index(str(vector_db_dir / config.INDEX_PATH.name))
    metadata = load_json(vector_db_dir / config.METADATA_PATH.name)
    return index, metadata


def validate_index(index: faiss.Index, metadata: list[dict], chunk_dir: Path) -> None:
    """Raise `StaleArtifactError` unless the index and metadata match the chunk files on disk."""
    if index.ntotal != len(metadata):
        raise StaleArtifactError(
            f"Index has {index.ntotal} vectors but metadata lists {len(metadata)} chunks. "
            "Rerun the indexing cell."
        )
    on_disk = {p.name for p in Path(chunk_dir).glob("*_chunk_*.txt")}
    indexed = {m["chunk_file"] for m in metadata}
    if on_disk != indexed:
        raise StaleArtifactError(
            f"Index does not match {chunk_dir}: {len(indexed - on_disk)} indexed chunks are missing on disk, "
            f"{len(on_disk - indexed)} chunks on disk are not indexed. Rerun the indexing cell."
        )
    required = {"title", "sections", "passage_types", "source_passages", "flat_char_start", "flat_char_end"}
    if any(not required.issubset(m) for m in metadata):
        raise StaleArtifactError("Index metadata lacks article structure. Rerun the indexing cell.")


def load_chunk_texts(metadata: list[dict], chunk_dir: Path) -> list[str]:
    """Chunk texts in metadata order."""
    return [(Path(chunk_dir) / m["chunk_file"]).read_text(encoding="utf-8") for m in metadata]
