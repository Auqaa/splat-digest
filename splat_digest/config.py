from __future__ import annotations

import os
from dataclasses import dataclass


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no dependency). Does not override existing env vars."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Config:
    deepseek_api_key: str
    telegram_bot_token: str
    telegram_chat_id: str
    deepseek_model: str = "deepseek-flash"
    deepseek_base_url: str = "https://api.deepseek.com"
    lookback_days: int = 3
    min_relevance: int = 5
    max_items_llm: int = 80
    llm_concurrency: int = 4
    state_path: str = "data/state.db"
    send_empty_digest: bool = True

    @classmethod
    def from_env(cls, require_secrets: bool = True) -> "Config":
        _load_dotenv()
        e = os.environ
        missing = [k for k in ("DEEPSEEK_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID") if not e.get(k)]
        if require_secrets and missing:
            raise SystemExit(f"Не заданы переменные окружения: {', '.join(missing)} (см. .env.example)")
        return cls(
            deepseek_api_key=e.get("DEEPSEEK_API_KEY", ""),
            telegram_bot_token=e.get("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=e.get("TELEGRAM_CHAT_ID", ""),
            deepseek_model=e.get("DEEPSEEK_MODEL") or cls.deepseek_model,
            deepseek_base_url=(e.get("DEEPSEEK_BASE_URL") or cls.deepseek_base_url).rstrip("/"),
            lookback_days=int(e.get("LOOKBACK_DAYS") or cls.lookback_days),
            min_relevance=int(e.get("MIN_RELEVANCE") or cls.min_relevance),
            max_items_llm=int(e.get("MAX_ITEMS_LLM") or cls.max_items_llm),
            llm_concurrency=int(e.get("LLM_CONCURRENCY") or cls.llm_concurrency),
            state_path=e.get("STATE_PATH") or cls.state_path,
            send_empty_digest=(e.get("SEND_EMPTY_DIGEST", "1") not in ("0", "false", "no")),
        )
