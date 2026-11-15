from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

KEYWORD_RE = re.compile(r"gaussian[\s-]*splat|\b3d\s*gs\b|\b3dgs\b|\bgsplat|\bsplatting\b", re.I)
ARXIV_ID_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf|html)/|^)(\d{4}\.\d{4,5})(?:v\d+)?", re.I)


def is_on_topic(*texts: str | None) -> bool:
    return any(t and KEYWORD_RE.search(t) for t in texts)


def arxiv_id_from(text: str | None) -> str | None:
    if not text:
        return None
    m = ARXIV_ID_RE.search(text.strip())
    return m.group(1) if m else None


@dataclass
class Item:
    """One article/post. `key` is the dedup key across sources."""

    key: str
    title: str
    url: str
    abstract: str = ""
    published: datetime | None = None
    authors: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)  # "arxiv", "hf", "hn"
    extra: dict = field(default_factory=dict)  # hf_upvotes, hn_points, hn_url, arxiv_comment, ...

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
        # arXiv URL is the canonical one; keep it if we have it.
        if "arxiv.org" not in self.url and "arxiv.org" in other.url:
            self.url = other.url
        for k, v in other.extra.items():
            self.extra.setdefault(k, v)


def make_key(url: str | None, fallback: str) -> str:
    aid = arxiv_id_from(url)
    return f"arxiv:{aid}" if aid else fallback
