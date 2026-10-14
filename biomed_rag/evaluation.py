"""Retrieval metrics, LLM-judge scoring and the Markdown evaluation report."""

import re
from pathlib import Path

from . import prompts
from .utils import StaleArtifactError, load_json

INVALID_GOLD = {"", "nan", "not specified", "n/a", "none", "no answer", "unknown"}


def gold_is_valid(ans) -> bool:
    s = str(ans).strip().lower()
    return s not in INVALID_GOLD and not s.startswith("not specified")


def retrieved_doc_ids(record: dict) -> list[str]:
    """Unique document IDs of the retrieved passages, in rank order."""
    doc_ids = []
    for passage in record["retrieved_passages"]:
        doc_id = str(passage["doc_id"]).strip().upper()
        if doc_id not in doc_ids:
            doc_ids.append(doc_id)
    return doc_ids


def retrieval_flags(record: dict) -> dict:
    """Whether the gold article or chunk appears in the final retrieved context."""
    gold_doc = str(record["doc_id"]).strip().upper()
    gold_chunk = str(record["chunk_file"]).strip().lower()

    doc_ids = retrieved_doc_ids(record)

    primary_chunk_files = [
        str(p["chunk_file"]).strip().lower()
        for p in record["retrieved_passages"]
    ]

    context_chunk_files = {
        str(chunk_file).strip().lower()
        for passage in record["retrieved_passages"]
        for chunk_file in passage.get(
            "chunk_files",
            [passage["chunk_file"]],
        )
    }

    return {
        "doc_hit": gold_doc in doc_ids,
        "doc_match_at_1": (
            bool(doc_ids)
            and doc_ids[0] == gold_doc
        ),
        "chunk_match_at_1": (
            bool(primary_chunk_files)
            and primary_chunk_files[0] == gold_chunk
        ),
        "chunk_hit": gold_chunk in context_chunk_files,
    }


def judge_context(record: dict) -> str:
    """Retrieved passages as '[chunk_file]: text' blocks separated by '---'."""
    context = "\n---\n".join(f"[{p['chunk_file']}]: {p['text']}" for p in record["retrieved_passages"])
    return context


def parse_judge_scores(raw_output: str) -> tuple[int | None, int | None]:
    c = re.search(r"<correctness>\s*([0-5])\s*</correctness>", raw_output)
    f = re.search(r"<faithfulness>\s*([0-5])\s*</faithfulness>", raw_output)
    if c and f:
        return int(c.group(1)), int(f.group(1))
    digits = [int(s) for s in re.findall(r'\b[0-5]\b', raw_output)]
    if len(digits) >= 2:
        return digits[0], digits[1]
    return None, None  # parse failure — not zero


def judge_record(judge, record: dict, max_new_tokens: int = 30) -> dict:
    """Score one RAG answer with the judge (an `LLM`, or anything with `generate(prompt, max_new_tokens)`)."""
    prompt = prompts.qwen_chat(prompts.JUDGE_TEMPLATE.format(
        question=record["question"],
        context=judge_context(record),
        gold_answer=record["gold_answer"],
        rag_answer=record["rag_answer"],
    ))
    raw_output = judge.generate(prompt, max_new_tokens=max_new_tokens)
    correctness, faithfulness = parse_judge_scores(raw_output)
    return {
        "correctness_score": correctness,
        "faithfulness_score": faithfulness,
        "judge_raw_output": raw_output,
    }


def evaluate(answers: list[dict], judge, max_new_tokens: int = 30) -> tuple[list[dict], dict]:
    """Add retrieval flags and judge scores to every RAG answer; return the records and the summary."""
    print("=== Starting Grounded Qwen Evaluation ===")
    records = []
    for number, answer in enumerate(answers, start=1):
        record = {
            **answer,
            **retrieval_flags(answer),
            "correctness_score": None,
            "faithfulness_score": None,
            "judge_raw_output": None,
        }
        if not gold_is_valid(answer["gold_answer"]):
            record["eval_status"] = "skipped_invalid_gold"
            print(f"Row {number} | SKIPPED | invalid gold answer: {str(answer['gold_answer'])!r}")
        else:
            record.update(judge_record(judge, answer, max_new_tokens))
            if record["correctness_score"] is None:
                record["eval_status"] = "parse_failure"
                print(f"Row {number} | PARSE FAILURE | raw: {record['judge_raw_output'][:60]!r}")
            else:
                record["eval_status"] = "scored"
                print(f"Row {number} | Correct: {record['correctness_score']}/5 | "
                      f"Faithful: {record['faithfulness_score']}/5 | "
                      f"Q: {str(answer['question'])[:40]}... | Gold: {str(answer['gold_answer'])[:30]}...")
        records.append(record)
    return records, summarize(records)


def _rate(count: int, total: int) -> dict:
    return {"count": count, "total": total, "rate": count / total if total else 0.0}


def summarize(records: list[dict]) -> dict:
    n = len(records)
    scored = [r for r in records if r["eval_status"] == "scored"]

    def numbers(status):
        return [i for i, r in enumerate(records, start=1) if r["eval_status"] == status]

    def mean(key):
        return sum(r[key] for r in scored) / len(scored) if scored else None

    return {
        "n_questions": n,
        "doc_hit_at_k": _rate(sum(r["doc_hit"] for r in records), n),
        "doc_match_at_1": _rate(sum(r["doc_match_at_1"] for r in records), n),
        "chunk_match_at_1": _rate(sum(r["chunk_match_at_1"] for r in records), n),
        "context_chunk_hit": _rate(sum(r["chunk_hit"] for r in records), n),
        "retrieval_refusals": sum(not r["retrieved_passages"] for r in records),
        "judge": {
            "n_scored": len(scored),
            "mean_correctness": mean("correctness_score"),
            "mean_faithfulness": mean("faithfulness_score"),
            "skipped_invalid_gold": numbers("skipped_invalid_gold"),
            "parse_failures": numbers("parse_failure"),
        },
    }


def _pct(metric: dict) -> str:
    return f"{metric['rate']:.2%} ({metric['count']}/{metric['total']})"


def _mean(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f} / 5"


def print_summary(summary: dict) -> None:
    judge = summary["judge"]
    print("\n=== FINAL EVALUATION METRICS ===")
    print(f"1. Document hit@k (gold article among retrieved passages): {_pct(summary['doc_hit_at_k'])}")
    print(f"2. Document match@1 (top passage from the gold article):    {_pct(summary['doc_match_at_1'])}")
    print(f"3. Exact chunk match@1 (top passage is the gold chunk):     {_pct(summary['chunk_match_at_1'])}")
    print(f"4. Context chunk hit (gold chunk anywhere in final context): {_pct(summary['context_chunk_hit'])}")
    print(f"5. Retrieval refusals (no passage above the threshold):     {summary['retrieval_refusals']}")
    print(f"6. Average Correctness Score (Qwen Judge):  {_mean(judge['mean_correctness'])}  (n={judge['n_scored']})")
    print(f"7. Average Faithfulness Score (Qwen Judge): {_mean(judge['mean_faithfulness'])}  (n={judge['n_scored']})")
    print(f"8. Skipped questions (missing/invalid gold answer): "
          f"{len(judge['skipped_invalid_gold'])} -> {judge['skipped_invalid_gold']}")
    print(f"9. Judge parse failures (excluded from averages): "
          f"{len(judge['parse_failures'])} -> {judge['parse_failures']}")


def _quote(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in str(text).split("\n"))


def _judge_text(record: dict) -> str:
    if record["eval_status"] == "scored":
        return f"correctness {record['correctness_score']}/5, faithfulness {record['faithfulness_score']}/5"
    if record["eval_status"] == "skipped_invalid_gold":
        return "skipped (invalid gold answer)"
    return f"parse failure (raw output: `{str(record['judge_raw_output'])[:80]}`)"


def write_markdown_report(records: list[dict], summary: dict, path: Path) -> Path:
    """Write a human-readable report: summary table, then every question with answers, scores and passages."""
    judge = summary["judge"]

    def yes_no(flag):
        return "yes" if flag else "no"

    lines = [
        "# RAG evaluation report",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Questions | {summary['n_questions']} |",
        f"| Document hit@k | {_pct(summary['doc_hit_at_k'])} |",
        f"| Document match@1 | {_pct(summary['doc_match_at_1'])} |",
        f"| Exact chunk match@1 | {_pct(summary['chunk_match_at_1'])} |",
        f"| Context chunk hit | {_pct(summary['context_chunk_hit'])} |",
        f"| Retrieval refusals | {summary['retrieval_refusals']} |",
        f"| Average correctness (n={judge['n_scored']}) | {_mean(judge['mean_correctness'])} |",
        f"| Average faithfulness (n={judge['n_scored']}) | {_mean(judge['mean_faithfulness'])} |",
        f"| Skipped (invalid gold answer) | {judge['skipped_invalid_gold'] or 'none'} |",
        f"| Judge parse failures | {judge['parse_failures'] or 'none'} |",
        "",
        "## Questions",
        "",
    ]
    for number, r in enumerate(records, start=1):
        lines += [
            f"### {number}. {r['question']}",
            "",
            f"- **Gold article / chunk:** `{r['doc_id']}` / `{r['chunk_file']}`",
            f"- **Retrieval:** document hit {yes_no(r['doc_hit'])}, document match@1 {yes_no(r['doc_match_at_1'])}, "
            f"primary chunk match@1 {yes_no(r['chunk_match_at_1'])}, "
            f"gold chunk in final context {yes_no(r['chunk_hit'])}",
            f"- **Judge:** {_judge_text(r)}",
            "",
            "**Gold answer:**",
            "",
            _quote(r["gold_answer"]),
            "",
            "**RAG answer:**",
            "",
            _quote(r["rag_answer"]),
            "",
        ]
        passages = r["retrieved_passages"]
        if passages:
            lines += [f"<details><summary>Retrieved passages ({len(passages)})</summary>", ""]
            for p in passages:
                members = ", ".join(f"`{c}`" for c in p["chunk_files"])
                sections = ", ".join(p.get("sections", [])) or "unknown"
                offsets = f"{p['source_offset_start']}-{p['source_offset_end']}" if p.get("source_offset_start") is not None else "unknown"
                lines += [
                    f"#### {p['rank']}. `{p['chunk_file']}` (score {p['score']:.3f})",
                    "",
                    f"Article: **{p.get('title') or p['doc_id']}** (`{p['doc_id']}`)",
                    "",
                    f"Section: {sections}; source offsets: {offsets}",
                    "",
                    f"Chunks: {members}",
                    "",
                    _quote(p["text"]),
                    "",
                ]
            lines += ["</details>", ""]
        else:
            lines += ["*No passages retrieved (best dense similarity below the threshold).*", ""]
        lines += ["---", ""]

    path = Path(path)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved evaluation report to '{path.name}'.")
    return path


def load_evaluation(path: Path, answers: list[dict]) -> tuple[list[dict], dict]:
    """Load saved evaluation results, checking they belong to the current RAG answers."""
    data = load_json(path)
    records, summary = data["records"], data["summary"]
    if [r["question"] for r in records] != [a["question"] for a in answers]:
        raise StaleArtifactError(
            f"{Path(path).name} does not match the current RAG answers. "
            "Rerun the evaluation cell."
        )
    return records, summary
