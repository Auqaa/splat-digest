"""LLM scoring via any OpenAI-compatible Chat Completions API (default: Kimi / Moonshot).

Kimi docs: https://platform.kimi.ai/docs/api/overview
- base_url https://api.moonshot.ai/v1, JSON mode via response_format=json_object.
- kimi-k2.6: thinking optional; temperature is fixed by the server (1.0 thinking / 0.6 non-thinking)
  and other values are rejected, so we never send `temperature`.
- kimi-k3*: always thinks, rejects the `thinking` field -> LLM_THINKING=omit (set automatically).
The configured model is validated against GET /models at startup.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

import httpx

from .models import Item

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ты — эксперт по 3D Gaussian Splatting (3DGS), нейронному рендерингу и фотограмметрии.
Ты оцениваешь новые записи (статьи, релизы инструментов, новости индустрии, посты сообщества) для инженера, которого интересует тема:
«3DGS для дронов и реконструкции интерьеров».

Что считается релевантным:
- Дроны/БПЛА: аэросъёмка, крупномасштабные/городские сцены, разреженные или наклонные ракурсы,
  неточные позы/GPS, motion blur, rolling shutter, онлайн/бортовая реконструкция, SLAM на 3DGS,
  навигация и планирование траектории съёмки (active view selection), сжатие для передачи.
- Интерьеры: комнаты/здания, бестекстурные стены, отражающие и прозрачные поверхности, освещение,
  извлечение меша/геометрии/нормалей, планировки, семантика и сегментация, съёмка с телефона/360°,
  few-shot интерьеры, LiDAR+камера, SLAM в помещениях.
- Общие улучшения 3DGS, которые напрямую применимы к этим сценариям (скорость обучения, качество геометрии,
  работа без COLMAP-поз, anti-aliasing, большие сцены, LOD) — средняя релевантность.

Шкала relevance (целое 0–10):
- 9–10: статья прямо про 3DGS для дронов/аэросъёмки ИЛИ для интерьеров.
- 7–8: 3DGS-метод, явно протестированный на аэро- или indoor-датасетах (Mill-19, UrbanScene3D, ScanNet,
  ScanNet++, Replica, Matterport3D и т.п.), либо SLAM/геометрия, сразу применимые.
- 4–6: общее улучшение 3DGS, полезное косвенно.
- 1–3: 3DGS, но другая область (аватары, люди, генерация объектов, медицина, автономные авто без переноса).
- 0: не про 3DGS / не статья / шум.
Не завышай оценки. Большинство статей должно получать 1–6.

Тип записи указан в поле «Тип»:
- paper: научная статья — оценивай по шкале выше.
- tool: релиз библиотеки/программы, новый репозиторий, модель/Space на Hugging Face. Оценивай практическую
  пользу для пайплайна «съёмка с дрона/в помещении → 3DGS → меш/просмотр». Крупный релиз gsplat/COLMAP/
  просмотрщика с важными фичами — 6–8; багфикс-релиз — 2–4; заброшенный/игрушечный репозиторий — 0–2.
- industry: новость компании/продукта (DJI, Pix4D, Matterport, XGRIDS и т.п.). 8–10 — продукт с 3DGS для
  дронов/интерьеров; 4–6 — смежное (фотограмметрия, LiDAR-сканеры) без 3DGS; 0–2 — маркетинг без сути.
- community: пост/обсуждение (Reddit, HN, Bluesky). Ценно: сравнения инструментов, практический опыт съёмки
  с дронов и в помещениях, воркфлоу. Шоукейс «посмотрите мой сплат» без деталей — 0–2.
Для не-статей is_paper=false.

Отвечай ТОЛЬКО валидным JSON-объектом такого вида:
{
  "relevance": 0,
  "relevance_reason": "почему такая оценка, 1 предложение по-русски",
  "summary_ru": "краткий пересказ на русском, 2–3 предложения: задача, метод, результат",
  "key_contribution": "главный вклад одной фразой по-русски",
  "drone_relevant": false,
  "indoor_relevant": false,
  "practical_value": "low | medium | high",
  "code_available": null,
  "tags": ["slam", "aerial", "indoor", "mesh", "compression", "..."],
  "is_paper": true
}
code_available: true, если в тексте явно указан код/репозиторий; false, если явно сказано, что кода нет; иначе null.
is_paper: false, если это не научная работа (например, пост-продукт, демо, новость).
Не выдумывай факты, которых нет в тексте. Если информации мало — так и напиши в summary_ru."""


@dataclass
class Analysis:
    relevance: int
    relevance_reason: str
    summary_ru: str
    key_contribution: str = ""
    drone_relevant: bool = False
    indoor_relevant: bool = False
    practical_value: str = "low"
    code_available: bool | None = None
    tags: list[str] = field(default_factory=list)
    is_paper: bool = True

    @classmethod
    def from_json(cls, data: dict) -> "Analysis":
        """Validate and coerce model output. Raises ValueError on unusable output."""
        if not isinstance(data, dict):
            raise ValueError("not an object")
        try:
            rel = int(round(float(data["relevance"])))
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"bad relevance: {data.get('relevance')!r}") from e
        summary = str(data.get("summary_ru") or "").strip()
        if not summary:
            raise ValueError("empty summary_ru")
        pv = str(data.get("practical_value") or "low").lower().strip()
        code = data.get("code_available")
        tags = data.get("tags") or []
        return cls(
            relevance=max(0, min(10, rel)),
            relevance_reason=str(data.get("relevance_reason") or "").strip(),
            summary_ru=summary,
            key_contribution=str(data.get("key_contribution") or "").strip(),
            drone_relevant=bool(data.get("drone_relevant")),
            indoor_relevant=bool(data.get("indoor_relevant")),
            practical_value=pv if pv in ("low", "medium", "high") else "low",
            code_available=code if isinstance(code, bool) else None,
            tags=[str(t) for t in tags if t][:6] if isinstance(tags, list) else [],
            is_paper=bool(data.get("is_paper", True)),
        )

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def build_user_prompt(item: Item) -> str:
    parts = [
        f"Тип: {item.kind}",
        f"Заголовок: {item.title}",
        f"Источники: {', '.join(item.sources)}",
        f"URL: {item.url}",
    ]
    if item.extra.get("venue"):
        parts.append(f"Площадка: {item.extra['venue']}")
    if item.extra.get("arxiv_category"):
        parts.append(f"Категория arXiv: {item.extra['arxiv_category']}")
    if item.extra.get("arxiv_comment"):
        parts.append(f"Комментарий авторов: {item.extra['arxiv_comment']}")
    parts.append(f"Аннотация/текст:\n{item.abstract or '(нет текста, только заголовок)'}")
    parts.append("Верни JSON.")
    return "\n".join(parts)


class LLMClient:
    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        model: str,
        base_url: str,
        thinking: str = "disabled",
        max_tokens: int = 1500,
    ):
        self.client = client
        self.model = model
        self.base_url = base_url
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

    async def check_model(self) -> None:
        """Fail fast with a clear message if the configured model ID does not exist."""
        r = await self.client.get(f"{self.base_url}/models", headers=self.headers, timeout=30)
        if r.status_code == 401:
            raise SystemExit("LLM: неверный LLM_API_KEY (401)")
        r.raise_for_status()
        ids = [m.get("id") for m in r.json().get("data", [])]
        if self.model not in ids:
            raise SystemExit(
                f"LLM: модели {self.model!r} нет в /models. Доступные: {ids}. Задайте LLM_MODEL."
            )
        log.info("LLM models: %s (using %s)", ids, self.model)

    async def analyze(self, item: Item, retries: int = 3) -> Analysis:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(item)},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_tokens,
        }
        if self.thinking == "disabled":
            # Thinking adds latency/cost and isn't needed for scoring an abstract.
            payload["thinking"] = {"type": "disabled"}
        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                r = await self.client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=self.headers, timeout=120
                )
                if r.status_code in (429, 500, 502, 503):
                    raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
                if r.status_code == 400 and "thinking" in r.text and "thinking" in payload:
                    # Older/other models may reject the param; drop it and retry.
                    payload.pop("thinking")
                    raise ValueError(f"400, retrying without thinking param: {r.text[:200]}")
                r.raise_for_status()
                data = r.json()
                for k in self.usage:
                    self.usage[k] += int((data.get("usage") or {}).get(k) or 0)
                content = data["choices"][0]["message"].get("content") or ""
                if not content.strip():
                    raise ValueError("empty content (max_tokens too low for a thinking model?)")
                return Analysis.from_json(json.loads(content))
            except (httpx.HTTPError, ValueError, KeyError, json.JSONDecodeError) as e:
                last_err = e
                log.warning("LLM attempt %d/%d failed for %s: %s", attempt, retries, item.key, e)
                await asyncio.sleep(min(2**attempt, 20))
        raise RuntimeError(f"LLM failed for {item.key}: {last_err}")
