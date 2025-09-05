# Content Pipeline (FastAPI + SQLAlchemy + Celery-ready)

- Config via `.env` (see `.env.example`).
- JSON-логирование на stdout.
- Идемпотентные операции (уникальные ограничения + проверки).
- Тесты: `pytest`; статическая проверка: `mypy`.

## Быстрый старт (Docker)

1) Скопируйте пример окружения: `.env.example` -> `.env` и задайте значения.
2) Соберите и запустите:

    docker compose build
    docker compose up -d db redis api

3) Откройте Swagger: http://localhost:8000/docs

## Переменные окружения (основные)
- DATABASE_URL — строка подключения к БД (по умолчанию формируется из настроек Postgres сервисов compose)
- JWT_SECRET — ключ для HS256 токенов (обязательно переопределить)
- LOG_LEVEL — уровень логирования
- Ключи провайдеров (DEEPL_API_KEY, GOOGLE_API_KEY, YANDEX_API_KEY, OPENAI_API_KEY) — задайте при использовании.

## Тесты и типы

    pip install -r requirements.txt
    pytest
    mypy

## Примечания
- В публикаторе VK загрузка фото упрощена (attachments=url). Для продакшена реализуйте upload+save.
- Cron-планировщик поддерживает базовые выражения для минут/часов; при необходимости подключите `croniter`.

