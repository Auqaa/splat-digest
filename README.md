# splat-digest

Раз в день собирает новые статьи и посты про 3D Gaussian Splatting, оценивает их через DeepSeek
по теме **«3DGS для дронов и реконструкции интерьеров»** и присылает дайджест в Telegram.

## Источники
| Источник | Как | Фильтр |
|---|---|---|
| arXiv | `export.arxiv.org/api/query`, `all:"gaussian splatting" OR abs:3DGS …`, сортировка по дате подачи | дата публикации за последние `LOOKBACK_DAYS` |
| HF Daily Papers | `huggingface.co/api/daily_papers?date=…` за каждый день окна | ключевые слова в заголовке/аннотации |
| Hacker News | Algolia `search_by_date`, `tags=story` | ключевые слова (Algolia допускает опечатки, поэтому есть повторная локальная проверка) |

Дубли склеиваются по arXiv ID: если статья есть на arXiv, в HF и на HN, это одна запись
со ссылками на все три источника (и с ▲ апвоутами HF / очками HN).

## Что возвращает LLM (JSON)
`relevance` (0–10), `relevance_reason`, `summary_ru`, `key_contribution`, `drone_relevant`,
`indoor_relevant`, `practical_value` (low/medium/high), `code_available`, `tags`, `is_paper`.
Шкала и критерии описаны в `splat_digest/llm.py` (`SYSTEM_PROMPT`). Поменяйте их под себя.

Запрос: `response_format: json_object`, thinking выключен (`"thinking": {"type": "disabled"}`),
temperature 0.2. Ответ валидируется, а при пустом или битом JSON делается повтор.

## Модель DeepSeek
По умолчанию `deepseek-flash`. Старые имена `deepseek-chat` и `deepseek-reasoner` выведены из эксплуатации
24.07.2026. При запуске бот запрашивает `GET /models` и, если модели нет, падает со списком доступных ID.
Другую модель (например, Pro) можно задать через `DEEPSEEK_MODEL`.

## Запуск локально
```bash
pip install -e .
cp .env.example .env   # заполнить ключи
splat-digest chat-id   # сначала написать боту /start: команда покажет ваш chat_id
splat-digest dry-run   # всё, кроме отправки: печатает дайджест в консоль
splat-digest run
```
Cron: `17 8 * * * cd /path/splat-digest && .venv/bin/splat-digest run >> digest.log 2>&1`

## Запуск в GitHub Actions (без сервера)
`.github/workflows/digest.yml` запускается ежедневно в 05:17 UTC.
- Secrets: `DEEPSEEK_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.
- Variables (необязательно): `DEEPSEEK_MODEL`, `MIN_RELEVANCE`.
- Состояние (SQLite) хранится в Actions cache.

## Состояние и надёжность
- Статья помечается отправленной только после того, как Telegram принял сообщение.
- Результаты LLM кешируются, поэтому при сбое отправки повторный запуск не платит второй раз.
- Если LLM не справился со статьёй, она не помечается и попадает в следующий запуск.
- Если упал один источник, дайджест всё равно уходит, а в шапке появляется предупреждение.
- Если новых статей нет, приходит короткое сообщение: так видно, что бот жив (`SEND_EMPTY_DIGEST=0`, чтобы отключить).

## Тесты
```bash
pip install -e '.[dev]' && pytest
```
