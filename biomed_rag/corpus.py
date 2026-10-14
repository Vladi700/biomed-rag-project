"""Downloading PubMed Central articles in BioC format, filtering sections and extracting plain text."""

import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config
from .utils import save_json


def build_pmc_query(days: int = 365, today: datetime | None = None) -> str:
    """PMC search query for open-access "machine learning" articles from the last `days` days."""
    today = today or datetime.now()
    # Restrict to the last 12 months, sorted by relevance
    end_date = today.strftime("%Y/%m/%d")
    start_date = (today - timedelta(days=days)).strftime("%Y/%m/%d")
    return (
        '"machine learning"[Title/Abstract] AND  '
        'open access[filter] '
        f'AND ("{start_date}"[PDAT] : "{end_date}"[PDAT])'
    )


def search_pmc_ids(query: str, max_results: int = 1000) -> list[str]:
    """Search PMC and return the matching article IDs, sorted by relevance."""
    params = {
        "db": config.DB,
        "term": query,
        "retmax": max_results,
        "retstart": 0,
        "sort": "relevance",
        "usehistory": "y",
        "email": config.EMAIL,
        "tool": config.TOOL,
        "retmode": "json"
    }
    r = requests.get(config.ESEARCH_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    ids = data["esearchresult"]["idlist"]
    return ids


def make_session() -> requests.Session:
    """HTTP session that retries transient server errors."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def fetch_bioc_xml(session: requests.Session, pmc_id: str, max_retries: int = 5) -> str | None:
    """Download one article as BioC XML; None if it is unavailable."""
    url = f"{config.BIOC_BASE}/{pmc_id}/unicode"
    headers = {"User-Agent": f"{config.TOOL} ({config.EMAIL})"}
    for attempt in range(max_retries):
        try:
            r = session.get(url, headers=headers, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"Request error for {pmc_id}: {e}")
            return None

        if r.status_code == 200:
            if "[Error]" in r.text or r.text.strip().startswith("<!DOCTYPE"):
                return None
            return r.text

        if r.status_code == 429:
            sleep_time = 2 ** attempt  # 1, 2, 4, 8, 16 s
            print(f"Rate limited for {pmc_id}, sleeping {sleep_time}s")
            time.sleep(sleep_time)
            continue

        print(f"HTTP {r.status_code} for {pmc_id}")
        return None
    return None


def download_corpus(pmc_ids: list[str], xml_dir: Path, n_articles: int = 500,
                    session: requests.Session | None = None, delay_s: float = 1.0) -> list[str]:
    """Download up to `n_articles` BioC XML files into `xml_dir`, skipping existing ones; return the failed IDs."""
    session = session or make_session()
    xml_dir = Path(xml_dir)
    xml_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    failed_xml = []

    for raw_id in pmc_ids:
        if downloaded >= n_articles:
            break
        pmc_id = f"PMC{raw_id}"
        out_path = xml_dir / f"{pmc_id}.xml"
        if out_path.exists():
            downloaded += 1
            continue

        xml_text = fetch_bioc_xml(session, pmc_id)
        if xml_text is None:
            failed_xml.append(pmc_id)
            print("Failed to fetch XML for", pmc_id)
            continue

        out_path.write_text(xml_text, encoding="utf-8")
        downloaded += 1
        print(f"Saved {pmc_id}.xml ({len(xml_text)} chars) [{downloaded}/{n_articles}]")

        time.sleep(delay_s)  # NCBI rate limit

    print(f"\nDone. Downloaded {downloaded} XML files to {xml_dir}/; failed {len(failed_xml)}.")
    return failed_xml


def count_section_types(xml_dir: Path) -> dict:
    """Count passages per section_type over all XML files in `xml_dir`."""
    xml_dir = Path(xml_dir)
    section_types = Counter()
    no_section = 0
    unparsed = 0

    xml_paths = sorted(xml_dir.glob("*.xml"))
    for xml_path in xml_paths:
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            unparsed += 1
            continue

        for p in root.findall(".//passage"):
            st_node = p.find("infon[@key='section_type']")
            if st_node is not None and st_node.text:
                section_types[st_node.text.strip().upper()] += 1
            else:
                no_section += 1

    return {
        "counts": section_types,
        "files_parsed": len(xml_paths) - unparsed,
        "unparsed": unparsed,
        "no_section": no_section,
    }


def print_section_counts(result: dict) -> None:
    """Print the output of `count_section_types`."""
    print(f"Files parsed: {result['files_parsed']}, unparsed: {result['unparsed']}")
    print(f"Passages without section_type: {result['no_section']}\n")
    print("Unique section types (with counts):")
    for st, count in result["counts"].most_common():
        print(f"  {st:20s}: {count}")


def _clear_outputs(out_dir: Path, pattern: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(pattern):
        old.unlink()


def filter_sections(in_dir: Path, out_dir: Path, excluded_sections: set[str],
                    excluded_types: set[str]) -> tuple[int, int]:
    """Write copies of the XML files without the excluded passages; return (kept, removed) passage counts."""
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    xml_paths = sorted(in_dir.glob("PMC*.xml"))
    if not xml_paths:
        raise FileNotFoundError(f"No PMC*.xml files in {in_dir}")
    _clear_outputs(out_dir, "PMC*.xml")

    removed = 0
    kept = 0

    for xml_path in xml_paths:
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue

        root = tree.getroot()

        for doc in root.findall("document"):
            for passage in doc.findall("passage"):
                infons = {
                    i.get("key"): (i.text or "").strip()
                    for i in passage.findall("infon")
                }
                section_type = infons.get("section_type", "").upper()
                p_type = infons.get("type", "").lower()

                if section_type in excluded_sections or p_type in excluded_types:
                    doc.remove(passage)
                    removed += 1
                else:
                    kept += 1

        out_path = out_dir / xml_path.name
        tree.write(out_path, encoding="utf-8", xml_declaration=True)

    print(f"Wrote {len(list(out_dir.glob('PMC*.xml')))} filtered XML files to {out_dir}/")
    print(f"Passages kept: {kept}, removed: {removed}")
    return kept, removed


def extract_article_from_bioc_xml(xml_text: str) -> dict | None:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None

    document = root.find(".//document")
    if document is None:
        return None

    passages = []
    parts = []
    flat_pos = 0
    title = None
    article_ids = {}
    for passage_idx, passage in enumerate(document.findall("passage")):
        text = (passage.findtext("text") or "").strip()
        if not text:
            continue
        infons = {i.get("key"): (i.text or "").strip() for i in passage.findall("infon")}
        section_type = infons.get("section_type", "").upper()
        passage_type = infons.get("type", "").lower()
        source_offset = int(passage.findtext("offset", "0"))
        if parts:
            flat_pos += 2
        flat_start = flat_pos
        flat_end = flat_start + len(text)
        passages.append({
            "passage_idx": passage_idx,
            "text": text,
            "section_type": section_type,
            "passage_type": passage_type,
            "source_offset_start": source_offset,
            "source_offset_end": source_offset + len(text),
            "flat_char_start": flat_start,
            "flat_char_end": flat_end,
        })
        if section_type == "TITLE" and title is None:
            title = text
        for key, value in infons.items():
            if key and key.startswith("article-id_") and value:
                article_ids.setdefault(key.removeprefix("article-id_"), value)
        parts.append(text)
        flat_pos = flat_end

    if not parts:
        return None
    return {
        "doc_id": (document.findtext("id") or article_ids.get("pmc") or "").strip(),
        "title": title or "",
        "article_ids": article_ids,
        "text": "\n\n".join(parts),
        "passages": passages,
    }


def extract_text_from_bioc_xml(xml_text: str) -> str | None:
    article = extract_article_from_bioc_xml(xml_text)
    return article["text"] if article else None


def extract_corpus_text(xml_dir: Path, out_dir: Path) -> int:
    """Write plain text and structural metadata for each filtered BioC article."""
    xml_dir, out_dir = Path(xml_dir), Path(out_dir)
    xml_paths = sorted(xml_dir.glob("PMC*.xml"))
    if not xml_paths:
        raise FileNotFoundError(f"No PMC*.xml files in {xml_dir}")
    _clear_outputs(out_dir, "PMC*.txt")
    _clear_outputs(out_dir, "PMC*.metadata.json")

    for xml_path in xml_paths:
        pmc_id = xml_path.stem
        article = extract_article_from_bioc_xml(xml_path.read_text(encoding="utf-8"))
        if not article:
            print(f"No text for {pmc_id}, skipping.")
            continue

        article["doc_id"] = article["doc_id"] or pmc_id
        (out_dir / f"{pmc_id}.txt").write_text(article.pop("text"), encoding="utf-8")
        save_json(article, out_dir / f"{pmc_id}.metadata.json")

    n_articles = len(list(out_dir.glob("PMC*.txt")))
    print(f"\nDone. {n_articles} articles in {out_dir}/")
    return n_articles
