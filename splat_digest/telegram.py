from __future__ import annotations

import asyncio
import logging

import httpx

log = logging.getLogger(__name__)
API = "https://api.telegram.org/bot{token}/{method}"
LIMIT = 4096


async def send_messages(client: httpx.AsyncClient, token: str, chat_id: str, chunks: list[str]) -> None:
    for i, text in enumerate(chunks):
        assert len(text) <= LIMIT, f"chunk {i} too long: {len(text)}"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": True},
        }
        for attempt in range(5):
            r = await client.post(API.format(token=token, method="sendMessage"), json=payload, timeout=30)
            if r.status_code == 429:
                wait = r.json().get("parameters", {}).get("retry_after", 5)
                log.warning("Telegram 429, sleeping %ss", wait)
                await asyncio.sleep(wait)
                continue
            if r.status_code != 200 or not r.json().get("ok"):
                raise RuntimeError(f"Telegram sendMessage failed: {r.status_code} {r.text[:300]}")
            break
        else:
            raise RuntimeError("Telegram: too many 429s")
        await asyncio.sleep(1.1)  # stay well under per-chat rate limit


async def print_chat_ids(client: httpx.AsyncClient, token: str) -> None:
    """Helper: write /start to the bot, then run `splat-digest chat-id`."""
    r = await client.get(API.format(token=token, method="getUpdates"), timeout=30)
    r.raise_for_status()
    seen = set()
    for u in r.json().get("result", []):
        msg = u.get("message") or u.get("channel_post") or {}
        chat = msg.get("chat") or {}
        if chat.get("id") and chat["id"] not in seen:
            seen.add(chat["id"])
            print(chat["id"], chat.get("type"), chat.get("username") or chat.get("title") or "")
    if not seen:
        print("Нет апдейтов. Напишите боту /start и повторите.")
