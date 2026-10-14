"""RAG answering: retrieved passages are placed in the system prompt of the generator LLM."""

from pathlib import Path

from . import config
from .llm import LLM, extract_tag
from .prompts import RAG_SYSTEM_TEMPLATE, format_retrieved_documents
from .retrieval import HybridRetriever
from .utils import StaleArtifactError, load_json


def build_rag_prompt(llm, query: str, hits: list[dict]) -> str:
    """Chat prompt with the retrieved passages as <docN> blocks in the system message and the query as the user message."""
    messages = [
        {"role": "system", "content": RAG_SYSTEM_TEMPLATE.format(documents=format_retrieved_documents(hits))},
        {"role": "user", "content": query},
    ]
    return llm.chat_prompt(messages)


class RAGPipeline:
    """Retriever plus generator LLM."""

    def __init__(self, retriever: HybridRetriever, llm: LLM):
        self.retriever = retriever
        self.llm = llm

    @classmethod
    def load(cls, retriever: HybridRetriever, model_id: str = config.GENERATOR_MODEL_ID) -> "RAGPipeline":
        """Load the generator model for an existing retriever."""
        return cls(retriever, LLM.load(model_id, trust_remote_code=True))

    def answer(self, query: str, k: int = 3, max_new_tokens: int = 100) -> dict:
        """Retrieve `k` passages and generate the answer; returns answer, raw output and hits."""
        hits = self.retriever.search(query, k=k)
        prompt = build_rag_prompt(self.llm, query, hits)
        raw_output = self.llm.generate(prompt, max_new_tokens=max_new_tokens)
        answer = extract_tag(raw_output, "answer")
        return {
            "answer": answer if answer is not None else raw_output.strip(),
            "raw_output": raw_output,
            "hits": hits,
        }

    def close(self) -> None:
        """Free the generator LLM (the retriever stays usable)."""
        self.llm.close()

    def __enter__(self) -> "RAGPipeline":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def answer_test_set(pipeline: RAGPipeline, test_set: list[dict], k: int = 3, max_new_tokens: int = 100) -> list[dict]:
    """Answer every test question; one record per question."""
    print(f"Generating RAG answers for {len(test_set)} questions...")
    records = []
    for idx, row in enumerate(test_set, 1):
        result = pipeline.answer(row["question"], k=k, max_new_tokens=max_new_tokens)
        records.append({
            "doc_id": row["doc_id"],
            "chunk_file": row["chunk_file"],
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "rag_answer": result["answer"],
            "retrieved_passages": result["hits"],
        })
        print(f"[{idx}/{len(test_set)}] Q: {row['question'][:50]}... -> Answered.")
    return records


def load_answers(path: Path, test_set: list[dict]) -> list[dict]:
    """Load the saved RAG answers; raise `StaleArtifactError` unless they answer the current test set."""
    answers = load_json(path)
    if [a["question"] for a in answers] != [t["question"] for t in test_set]:
        raise StaleArtifactError(
            f"{path} does not match the current test set. Rerun the RAG answering cell."
        )
    return answers
