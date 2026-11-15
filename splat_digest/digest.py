from __future__ import annotations

from datetime import date
from html import escape

from .llm import Analysis
from .models import COMMUNITY, INDUSTRY, PAPER, TOOL, Item
from .telegram import LIMIT

PV = {"high": "высокая", "medium": "средняя", "low": "низкая"}
SECTIONS = [
    (PAPER, "📄 Статьи"),
    (TOOL, "🛠 Инструменты и код"),
    (INDUSTRY, "🏭 Индустрия и продукты"),
    (COMMUNITY, "💬 Сообщество"),
]
# (extra key for url, label, extra key for counter, counter symbol)
LINKS = [
    ("code_url", "код", "gh_stars", "★"),
    ("hf_url", "HF", "hf_upvotes", "▲"),
    ("s2_url", "S2", "citations", "цит."),
    ("hn_url", "HN", "hn_points", "▲"),
    ("reddit_url", "Reddit", None, ""),
    ("bsky_url", "Bluesky", "bsky_likes", "♥"),
]


def _bar(score: int) -> str:
    return "🟩" * (score // 2) + "⬜" * (5 - score // 2)


def render_item(item: Item, a: Analysis) -> str:
    flags = []
    if a.drone_relevant:
        flags.append("🚁 дроны")
    if a.indoor_relevant:
        flags.append("🏠 интерьеры")
    if a.code_available or item.extra.get("code_url"):
        flags.append("💻 код")
    if item.extra.get("venue"):
        flags.append(f"🏛 {escape(item.extra['venue'][:60])}")
    links = [f'<a href="{escape(item.url, quote=True)}">источник</a>']
    for url_key, label, cnt_key, sym in LINKS:
        u = item.extra.get(url_key)
        if not u or u == item.url:
            continue
        cnt = item.extra.get(cnt_key) if cnt_key else None
        links.append(f'<a href="{escape(u, quote=True)}">{label}{f" {sym}{cnt}" if cnt else ""}</a>')

    lines = [
        f"<b>{a.relevance}/10</b> {_bar(a.relevance)}  <b>{escape(item.title)}</b>",
        escape(a.summary_ru),
    ]
    if a.key_contribution:
        lines.append(f"<i>Вклад:</i> {escape(a.key_contribution)}")
    if a.relevance_reason:
        lines.append(f"<i>Почему {a.relevance}:</i> {escape(a.relevance_reason)}")
    meta = [f"польза: {PV.get(a.practical_value, a.practical_value)}"] + flags
    if a.tags:
        meta.append(" ".join("#" + escape(t.replace(" ", "_").replace("-", "_")) for t in a.tags))
    lines.append(" · ".join(meta))
    lines.append(" | ".join(links))
    return "\n".join(lines)


def build_digest(
    scored: list[tuple[Item, Analysis]],
    min_relevance: int,
    day: date,
    failed: int = 0,
    notes: list[str] | None = None,
) -> list[str]:
    scored = sorted(scored, key=lambda p: (-p[1].relevance, p[0].title))
    top = [p for p in scored if p[1].relevance >= min_relevance]
    rest = [p for p in scored if p[1].relevance < min_relevance]

    header = (
        f"🧊 <b>3DGS-дайджест · {day.strftime('%d.%m.%Y')}</b>\n"
        f"Новых: {len(scored)} · релевантных (≥{min_relevance}): {len(top)}"
        + (f" · не обработано LLM: {failed}" if failed else "")
    )
    for n in notes or []:
        header += f"\n⚠️ {escape(n)}"
    blocks = [header]
    for kind, title in SECTIONS:
        sec = [p for p in top if p[0].kind == kind]
        if not sec:
            continue
        rendered = [render_item(i, a) for i, a in sec]
        # Glue the section title to its first item so a message never ends with a bare title.
        blocks.append(f"<b>{title} ({len(sec)})</b>\n\n{rendered[0]}")
        blocks += rendered[1:]
    if rest:
        tail = [f"<b>Ниже порога ({len(rest)}):</b>"]
        for i, a in rest[:25]:
            icon = dict(zip((PAPER, TOOL, INDUSTRY, COMMUNITY), "📄🛠🏭💬")).get(i.kind, "")
            tail.append(f'{a.relevance} {icon} <a href="{escape(i.url, quote=True)}">{escape(i.title[:120])}</a>')
        if len(rest) > 25:
            tail.append(f"…и ещё {len(rest) - 25}")
        blocks.append("\n".join(tail))
    return pack(blocks)


def pack(blocks: list[str], limit: int = LIMIT - 96) -> list[str]:
    """Pack blocks into Telegram-sized messages without splitting a block (unless it alone is too big)."""
    chunks: list[str] = []
    cur = ""
    for b in blocks:
        if len(b) > limit:  # pathological: split by lines
            for line in b.split("\n"):
                line = line[:limit]
                if len(cur) + len(line) + 1 > limit:
                    chunks.append(cur)
                    cur = ""
                cur = f"{cur}\n{line}" if cur else line
            continue
        sep = "\n\n" if cur else ""
        if len(cur) + len(sep) + len(b) > limit:
            chunks.append(cur)
            cur = b
        else:
            cur = cur + sep + b
    if cur:
        chunks.append(cur)
    return chunks
