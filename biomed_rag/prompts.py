"""Prompt templates for test-set generation, RAG answering and LLM judging."""

QUESTION_GENERATION_TEMPLATE = """BACKGROUND
You are a student trying to get answers from a RAG system by asking it broad questions. You do NOT have any context about the underlying study, the paper, the dataset, the specific analysis, or the database from which your information comes.

YOUR TASK
I will provide you with a passage of a Pubmed article. Your task is to generate 1 clear, direct question based on this text and frame it as you have never seen the text.

CONSTRAINTS
- Frame the question as a general student query. Ask about the tool, metric, gene, method, or concept broadly.
- You do NOT know that there is a 'study', 'dataset', 'paper', 'analysis', or 'author'. Never mention these words or refer to them.
- DO NOT refer to the text or creators of the text (do not use words like 'the passage', 'this study', 'the author', 'the paper', 'here', 'the dataset', 'the analysis').
- Do not output any answer, commentary, or explanation.
- Make sure the answear is contained in the passage.

OUTPUT FORMAT
Output the question wrapped inside tags: <question>Your generated question</question>

PASSAGE:
{passage}
"""

GOLD_ANSWER_TEMPLATE = """BACKGROUND
You are a precise biomedical question-answering assistant that answears from it's own knowledge. You do NOT have any context about the underlying study, the paper, the dataset, the specific analysis, or the database from which your information comes.

YOUR TASK
Answer the question using ONLY the provided passage.

CONSTRAINTS
- Answer using ONLY facts directly mentioned in the passage.
- Do not say 'Based on the passage' or 'According to the text'.
- Do not output any commentary.

OUTPUT FORMAT
Output the concise answer wrapped inside tags: <answer>Your concise answer</answer>

PASSAGE:
{passage}

QUESTION:
{question}
"""

JUDGE_TEMPLATE = """BACKGROUND
You are an objective, precise AI evaluator grading a RAG system's answer to a biomedical question.

YOUR TASK
You are given the QUESTION, the RETRIEVED CONTEXT the system used, the GOLD ANSWER, and the RAG SYSTEM ANSWER.
Produce TWO scores, each from 0 to 5:

1) CORRECTNESS — how well the RAG SYSTEM ANSWER answers the QUESTION, judged against the GOLD ANSWER *and* the RETRIEVED CONTEXT. An answer may score 4-5 even if worded differently from the gold answer, but only if it is actually supported by the RETRIEVED CONTEXT or matches the gold answer. Do NOT reward plausible-sounding claims that are not in the context or gold answer.

2) FAITHFULNESS — whether every factual claim in the RAG SYSTEM ANSWER is supported by the RETRIEVED CONTEXT.
- 5: every claim is directly supported by the context
- 4: supported with trivially minor additions
- 3: mostly supported, one unsupported minor claim
- 2: several unsupported claims
- 1: mostly unsupported
- 0: contradicts the context or entirely unsupported

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

GOLD ANSWER:
{gold_answer}

RAG SYSTEM ANSWER:
{rag_answer}

CONSTRAINTS
- Provide only the two scores, no explanation.

OUTPUT FORMAT
<correctness>NUMBER</correctness>
<faithfulness>NUMBER</faithfulness>
"""

RAG_SYSTEM_TEMPLATE = """The following documents may provide information relevant to the user's query:
{documents}

Answer using only the supplied documents. Cite each supported claim with its source label, such as [doc1]. Do not invent citations."""

NO_DOCUMENTS = "(no documents retrieved)"


def qwen_chat(user_msg: str) -> str:
    """Wrap a user message in the ChatML format used for the Qwen prompts."""
    return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"


def format_documents(texts: list[str]) -> str:
    """Wrap retrieved passages in the <docN> blocks expected by LFM2-1.2B-RAG."""
    if not texts:
        return NO_DOCUMENTS
    return "\n".join(f"<doc{i}>\n{text}\n</doc{i}>" for i, text in enumerate(texts, start=1))


def format_retrieved_documents(hits: list[dict]) -> str:
    """Format retrieved text with stable source labels and article-level citation metadata."""
    documents = []
    for i, hit in enumerate(hits, start=1):
        header = [f"Source label: [doc{i}]", f"PMC ID: {hit['doc_id']}"]
        if hit.get("title"):
            header.append(f"Article title: {hit['title']}")
        if hit.get("sections"):
            header.append(f"Section: {', '.join(hit['sections'])}")
        start, end = hit.get("source_offset_start"), hit.get("source_offset_end")
        if start is not None and end is not None:
            header.append(f"Source offsets: {start}-{end}")
        documents.append("\n".join([*header, "", hit["text"]]))
    return format_documents(documents)
