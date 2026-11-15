from __future__ import annotations

from datetime import date
from html import escape

from .llm import Analysis
from .models import Item
from .telegram import LIMIT

PV = {"high": "высокая", "medium": "средняя", "low": "низкая"}


def _bar(score: int) -> str:
    return "🟩" * (score // 2) + "⬜" * (5 - score // 2)


def render_item(item: Item, a: Analysis) -> str:
    flags = []
    if a.drone_relevant:
        flags.append("🚁 дроны")
    if a.indoor_relevant:
        flags.append("🏠 интерьеры")
    if a.code_available:
        flags.append("💻 код")
    if not a.is_paper:
        flags.append("📰 не статья")
    links = [f'<a href="{escape(item.url, quote=True)}">источник</a>']
    if item.extra.get("hf_url"):
        up = item.extra.get("hf_upvotes")
        links.append(f'<a href="{escape(item.extra["hf_url"], quote=True)}">HF{f" ▲{up}" if up else ""}</a>')
    if item.extra.get("hn_url"):
        pts = item.extra.get("hn_points")
        links.append(f'<a href="{escape(item.extra["hn_url"], quote=True)}">HN{f" ▲{pts}" if pts else ""}</a>')

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
    blocks += [render_item(i, a) for i, a in top]
    if rest:
        tail = [f"<b>Ниже порога ({len(rest)}):</b>"]
        for i, a in rest[:25]:
            tail.append(f'{a.relevance} · <a href="{escape(i.url, quote=True)}">{escape(i.title[:120])}</a>')
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
