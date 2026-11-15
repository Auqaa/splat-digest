from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

KEYWORD_RE = re.compile(r"gaussian[\s-]*splat|\b3d\s*gs\b|\b3dgs\b|\bgsplat|\bsplatting\b", re.I)
ARXIV_ID_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf|html)/|arxiv:\s*|^)(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
ARXIV_ANY_RE = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})", re.I)

# Kinds drive the digest sections and the LLM instructions.
PAPER, TOOL, INDUSTRY, COMMUNITY = "paper", "tool", "industry", "community"
KIND_PRIORITY = {PAPER: 0, TOOL: 1, INDUSTRY: 2, COMMUNITY: 3}


def is_on_topic(*texts: str | None, pattern: re.Pattern = KEYWORD_RE) -> bool:
    return any(t and pattern.search(t) for t in texts)


def arxiv_id_from(text: str | None) -> str | None:
    if not text:
        return None
    m = ARXIV_ID_RE.search(text.strip())
    return m.group(1) if m else None


def arxiv_ids_in(text: str | None) -> list[str]:
    return list(dict.fromkeys(ARXIV_ANY_RE.findall(text or "")))


def norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.lower())


@dataclass
class Item:
    """One article/post/release. `key` is the dedup key across sources."""

    key: str
    title: str
    url: str
    abstract: str = ""
    published: datetime | None = None
    authors: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    kind: str = PAPER
    extra: dict = field(default_factory=dict)

    def merge(self, other: "Item") -> None:
        for s in other.sources:
            if s not in self.sources:
                self.sources.append(s)
        if len(other.abstract) > len(self.abstract):
            self.abstract = other.abstract
        if not self.authors and other.authors:
            self.authors = other.authors
        if self.published is None:
            self.published = other.published
        if "arxiv.org" not in self.url and "arxiv.org" in other.url:
            self.url = other.url
        if KIND_PRIORITY.get(other.kind, 9) < KIND_PRIORITY.get(self.kind, 9):
            self.kind = other.kind
        for k, v in other.extra.items():
            if v not in (None, "", []):
                self.extra.setdefault(k, v)


def make_key(url: str | None, fallback: str, text: str | None = None) -> str:
    aid = arxiv_id_from(url) or (arxiv_ids_in(text)[:1] or [None])[0]
    return f"arxiv:{aid}" if aid else fallback
