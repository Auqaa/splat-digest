from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field


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


ALL_SOURCES = (
    "arxiv_rss", "arxiv", "hf_daily", "hn", "s2", "openreview", "github_releases", "github_repos",
    "awesome", "hf_hub", "reddit", "bluesky", "vendors",
)


# Provider presets; any field can still be overridden with LLM_* env vars.
PROVIDERS = {
    # https://api-docs.deepseek.com/quick_start/pricing : thinking is ON by default -> disable it.
    "deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-flash", "thinking": "disabled"},
    # https://platform.kimi.ai/docs/api/overview
    "kimi": {"base_url": "https://api.moonshot.ai/v1", "model": "kimi-k2.6", "thinking": "disabled"},
}


@dataclass(frozen=True)
class Config:
    llm_api_key: str
    telegram_bot_token: str
    telegram_chat_id: str
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-flash"
    llm_base_url: str = "https://api.deepseek.com"
    # "disabled" -> send {"thinking":{"type":"disabled"}}; "omit" -> don't send the field.
    llm_thinking: str = "disabled"
    llm_max_tokens: int = 1500
    lookback_days: int = 3
    min_relevance: int = 5
    max_items_llm: int = 150
    llm_concurrency: int = 4
    state_path: str = "data/state.db"
    send_empty_digest: bool = True
    # >0: skip the run if a digest was delivered within this many hours (backup cron triggers).
    min_hours_between: float = 0
    sources: tuple[str, ...] = ALL_SOURCES
    sources_file: str = "sources.toml"
    github_token: str = ""
    s2_api_key: str = ""
    bluesky_handle: str = ""
    bluesky_app_password: str = ""
    # Second pass: model reads the full paper text (DeepSeek `deepseek_read_url` server tool).
    deep_read: bool = True
    deep_read_min: int = 7
    deep_read_max: int = 5
    sources_conf: dict = field(default_factory=dict, compare=False)

    @classmethod
    def from_env(cls, require_secrets: bool = True) -> "Config":
        _load_dotenv()
        e = os.environ
        provider = (e.get("LLM_PROVIDER") or cls.llm_provider).lower()
        if provider not in PROVIDERS:
            raise SystemExit(f"LLM_PROVIDER={provider!r}: допустимо {list(PROVIDERS)}")
        preset = PROVIDERS[provider]
        api_key = e.get("LLM_API_KEY") or e.get(f"{provider.upper()}_API_KEY", "")
        missing = [k for k, v in (("LLM_API_KEY", api_key), ("TELEGRAM_BOT_TOKEN", e.get("TELEGRAM_BOT_TOKEN")),
                                  ("TELEGRAM_CHAT_ID", e.get("TELEGRAM_CHAT_ID"))) if not v]
        if require_secrets and missing:
            raise SystemExit(f"Не заданы переменные окружения: {', '.join(missing)} (см. .env.example)")
        sources_file = e.get("SOURCES_FILE") or cls.sources_file
        conf = {}
        if os.path.exists(sources_file):
            with open(sources_file, "rb") as f:
                conf = tomllib.load(f)
        srcs = tuple(s.strip() for s in (e.get("SOURCES") or ",".join(ALL_SOURCES)).split(",") if s.strip())
        unknown = set(srcs) - set(ALL_SOURCES)
        if unknown:
            raise SystemExit(f"Неизвестные источники в SOURCES: {sorted(unknown)}. Доступны: {ALL_SOURCES}")
        model = e.get("LLM_MODEL") or preset["model"]
        is_k3 = model.startswith("kimi-k3")
        return cls(
            llm_api_key=api_key,
            llm_provider=provider,
            telegram_bot_token=e.get("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=e.get("TELEGRAM_CHAT_ID", ""),
            llm_model=model,
            llm_base_url=(e.get("LLM_BASE_URL") or preset["base_url"]).rstrip("/"),
            # K3 series always thinks and rejects the `thinking` field.
            llm_thinking=e.get("LLM_THINKING") or ("omit" if is_k3 else preset["thinking"]),
            llm_max_tokens=int(e.get("LLM_MAX_TOKENS") or (8000 if is_k3 else cls.llm_max_tokens)),
            lookback_days=int(e.get("LOOKBACK_DAYS") or cls.lookback_days),
            min_relevance=int(e.get("MIN_RELEVANCE") or cls.min_relevance),
            max_items_llm=int(e.get("MAX_ITEMS_LLM") or cls.max_items_llm),
            llm_concurrency=int(e.get("LLM_CONCURRENCY") or cls.llm_concurrency),
            state_path=e.get("STATE_PATH") or cls.state_path,
            send_empty_digest=(e.get("SEND_EMPTY_DIGEST", "1") not in ("0", "false", "no")),
            min_hours_between=float(e.get("MIN_HOURS_BETWEEN") or 0),
            sources=srcs,
            sources_file=sources_file,
            github_token=e.get("GITHUB_TOKEN", ""),
            s2_api_key=e.get("S2_API_KEY", ""),
            bluesky_handle=e.get("BLUESKY_HANDLE", ""),
            bluesky_app_password=e.get("BLUESKY_APP_PASSWORD", ""),
            # read_url is a DeepSeek-only server tool.
            deep_read=provider == "deepseek" and e.get("DEEP_READ", "1") not in ("0", "false", "no"),
            deep_read_min=int(e.get("DEEP_READ_MIN") or cls.deep_read_min),
            deep_read_max=int(e.get("DEEP_READ_MAX") or cls.deep_read_max),
            sources_conf=conf,
        )
