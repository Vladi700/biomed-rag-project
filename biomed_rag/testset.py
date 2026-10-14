"""Synthetic test set: questions and gold answers generated from sampled chunks."""

import random
from pathlib import Path

from . import prompts
from .llm import extract_tag
from .utils import StaleArtifactError, load_json


def sample_chunk_paths(chunk_dir: Path, n_passages: int = 20, seed: int = 42) -> list[Path]:
    """Reproducible random sample of chunk files (sorted first, so the order does not depend on the filesystem)."""
    chunk_paths = sorted(Path(chunk_dir).glob("*_chunk_*.txt"))
    return random.Random(seed).sample(chunk_paths, n_passages)


def generate_question(llm, passage: str, max_new_tokens: int = 150) -> str | None:
    """Ask the LLM for one question about `passage`; None if the output has no <question> tag."""
    prompt = prompts.qwen_chat(prompts.QUESTION_GENERATION_TEMPLATE.format(passage=passage))
    return extract_tag(llm.generate(prompt, max_new_tokens), "question")


def generate_questions(llm, chunk_dir: Path, n_passages: int = 20, seed: int = 42) -> list[dict]:
    """Generate one question for each sampled chunk, skipping outputs without a <question> tag."""
    sampled_paths = sample_chunk_paths(chunk_dir, n_passages, seed)
    print(f"Sampled {len(sampled_paths)} passages.")

    records = []
    for idx, path in enumerate(sampled_paths, 1):
        passage = path.read_text(encoding="utf-8")
        question = generate_question(llm, passage)
        if question is None:
            print(f"[{idx}/{len(sampled_paths)}] No <question> tag in the output, skipping {path.name}.")
            continue
        records.append({
            "doc_id": path.name.split("_chunk_")[0],
            "chunk_file": path.name,
            "passage": passage,
            "question": question,
        })
        print(f"[{idx}/{len(sampled_paths)}] Generated Q: {question[:80]}...")
    return records


def generate_gold_answer(llm, passage: str, question: str, max_new_tokens: int = 100) -> str:
    """Answer `question` from `passage`; falls back to the whole output if the <answer> tag is missing."""
    prompt = prompts.qwen_chat(prompts.GOLD_ANSWER_TEMPLATE.format(passage=passage, question=question))
    raw_output = llm.generate(prompt, max_new_tokens)
    answer = extract_tag(raw_output, "answer")
    # Fallback to entire clean response if tags are missing
    return answer if answer is not None else raw_output.strip()


def generate_gold_answers(llm, records: list[dict]) -> list[dict]:
    """Return copies of `records` with a `gold_answer` added."""
    test_set = []
    for idx, record in enumerate(records, 1):
        answer = generate_gold_answer(llm, record["passage"], record["question"])
        test_set.append({**record, "gold_answer": answer})
        print(f"[{idx}/{len(records)}] Q: {record['question'][:50]}... -> A: {answer}")
    return test_set


def load_test_set(path: Path, chunk_dir: Path) -> list[dict]:
    """Load the saved test set; raise `StaleArtifactError` if its chunks are no longer in `chunk_dir`."""
    test_set = load_json(path)
    missing = [r["chunk_file"] for r in test_set if not (Path(chunk_dir) / r["chunk_file"]).exists()]
    if missing:
        raise StaleArtifactError(
            f"{len(missing)} chunks of the saved test set are missing from {chunk_dir} (e.g. {missing[0]}). "
            "Rerun the test-set generation cells."
        )
    return test_set
