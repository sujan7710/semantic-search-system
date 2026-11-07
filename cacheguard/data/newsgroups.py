"""Loading and cleaning the 20 Newsgroups corpus.

The original preprocessing removed only the From:/Subject: lines, so routing
headers (Xref, Path, Newsgroups) stayed in every document. Those headers name
the newsgroup, which leaked the label into the embedding and inflated
retrieval quality. They also used up much of MiniLM's 256-token window.
This version drops the whole header block, quoted replies, and signatures.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Bump when cleaning changes so stale embeddings are detected at load time.
PREPROCESSING_VERSION = "2"

_QUOTE_LINE = re.compile(
    r"^\s*(>|\|)|^\s*In article\b|(writes|wrote|says|said)\s*:\s*$|writes in\b",
    re.IGNORECASE,
)
_SIGNATURE_DELIMITER = re.compile(r"^\s*-{2,}\s*$")
_EMAIL = re.compile(r"\S+@\S+")
_URL = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def strip_headers(raw: str) -> str:
    """Drop the RFC-822 style header block (everything before the first blank line)."""
    _, sep, body = raw.replace("\r\n", "\n").partition("\n\n")
    return body if sep else ""


def strip_quotes(body: str) -> str:
    """Drop quoted reply lines and 'In article ... writes:' attributions."""
    return "\n".join(line for line in body.split("\n") if not _QUOTE_LINE.search(line))


def strip_signature(body: str, max_signature_lines: int = 15) -> str:
    """Cut at a '--' delimiter line if it appears near the end of the post."""
    lines = body.rstrip().split("\n")
    for i in range(len(lines) - 1, max(-1, len(lines) - 1 - max_signature_lines), -1):
        if _SIGNATURE_DELIMITER.match(lines[i]):
            return "\n".join(lines[:i])
    return body


def clean_text(raw: str) -> str:
    """Turn a raw newsgroup file into the text that gets embedded.

    Punctuation is kept on purpose: transformer tokenizers use it, unlike the
    bag-of-words models the original regex cleanup was written for.
    """
    body = strip_signature(strip_quotes(strip_headers(raw)))
    body = _EMAIL.sub(" ", body)
    body = _URL.sub(" ", body)
    return _WHITESPACE.sub(" ", body).strip()


@dataclass(frozen=True)
class Document:
    doc_id: int
    newsgroup: str
    text: str
    source_path: str


@dataclass
class LoadStats:
    files_read: int = 0
    unreadable: int = 0
    too_short: int = 0
    duplicates: int = 0
    per_newsgroup: dict[str, int] = field(default_factory=dict)


def load_newsgroups(
    data_dir: str | Path, min_chars: int = 20, dedupe: bool = True
) -> tuple[list[Document], LoadStats]:
    """Load, clean, filter and de-duplicate the corpus.

    Cross-posted messages appear once per newsgroup in the raw data; with
    ``dedupe`` only the first copy (in sorted folder order) is kept.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"dataset folder not found: {data_dir}")

    stats = LoadStats()
    documents: list[Document] = []
    seen: set[str] = set()

    for group_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        kept = 0
        for path in sorted(p for p in group_dir.iterdir() if p.is_file()):
            try:
                # latin-1 maps every byte, so decoding itself cannot fail.
                raw = path.read_text(encoding="latin-1")
            except OSError as exc:
                logger.warning("skipping unreadable file %s: %s", path, exc)
                stats.unreadable += 1
                continue
            stats.files_read += 1
            text = clean_text(raw)
            if len(text) < min_chars:
                stats.too_short += 1
                continue
            if dedupe:
                if text in seen:
                    stats.duplicates += 1
                    continue
                seen.add(text)
            documents.append(Document(len(documents), group_dir.name, text, str(path)))
            kept += 1
        stats.per_newsgroup[group_dir.name] = kept

    return documents, stats
