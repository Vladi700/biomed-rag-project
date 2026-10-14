"""Paths, model IDs and settings shared by the pipeline."""

import os
from pathlib import Path

BASE_DIR = Path(os.environ.get("BIOMED_RAG_BASE_DIR", Path.cwd()))

XML_DIR = BASE_DIR / "pmc_xml_test"
FILTERED_XML_DIR = BASE_DIR / "pmc_xml_filtered"  # cleaned XMLs with excluded sections removed
RAW_TEXT_DIR = BASE_DIR / "pmc_raw_texts"
CHUNK_DIR = BASE_DIR / "document_repository"
VECTOR_DB_DIR = BASE_DIR / "vector_database"
INDEX_PATH = VECTOR_DB_DIR / "document_index.faiss"
METADATA_PATH = VECTOR_DB_DIR / "chunk_metadata.json"

PMC_IDS_PATH = BASE_DIR / "pmc_ids.json"
PROFILE_PATH = BASE_DIR / "dataset_profile.json"
TEST_SET_PATH = BASE_DIR / "micro_test_set.json"
RAG_ANSWERS_PATH = BASE_DIR / "lfm_rag_answers.json"
EVALUATION_PATH = BASE_DIR / "lfm_rag_answers_evaluated.json"
REPORT_PATH = BASE_DIR / "evaluation_report.md"

EMBEDDING_MODEL_ID = "NeuML/pubmedbert-base-embeddings"
GENERATOR_MODEL_ID = "LiquidAI/LFM2-1.2B-RAG"
QA_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
RERANKER_MODEL_ID = "BAAI/bge-reranker-base"

EMAIL = os.environ.get("NCBI_EMAIL", "your-email@example.com")
TOOL = "my_pmc_downloader"
DB = "pmc"
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
BIOC_BASE = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml"

EXCLUDED_SECTIONS = {
    "ABBR", "ACK_FUND", "APPENDIX", "AUTH_CONT",
    "COMP_INT", "REVIEW_INFO", "SUPPL", "REF",
}
EXCLUDED_TYPES = {"table"}  # raw table markup only
