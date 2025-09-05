"""
Translation task with provider adapters (DeepL, Google, Yandex).

- Celery task: translate_article(article_id: int, lang_target: str)
- Provider selection via LLM_PROVIDER env var
- Retries on HTTP errors, 15s timeout, input length limit
- Persists into translations table (SQLAlchemy ORM)
- Returns translation_id (idempotent w.r.t. (article_id, lang, provider))
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Protocol

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint, select
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship

from app.tasks.ingest import Article, Base, create_session_from_env

logger = logging.getLogger("app.tasks.translate")


# ------------------------------- ORM: Translation -------------------------------


class Translation(Base):
    __tablename__ = "translations"
    __table_args__ = (
        UniqueConstraint("article_id", "lang", "provider", name="uq_translation_article_lang_provider"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id"), nullable=False, index=True)
    lang: Mapped[str] = mapped_column(String(16), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    text: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    article: Mapped[Article] = relationship(Article, lazy="joined")


# --------------------------------- Interfaces ----------------------------------


class Requester(Protocol):
    def post_json(self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float) -> dict[str, Any]:
        ...


class HttpxRequester:
    """Requester backed by httpx.Client with per-call timeout."""

    def __init__(self) -> None:
        self._httpx = None
        try:
            import httpx as _httpx  # type: ignore

            self._httpx = _httpx
        except Exception:  # noqa: BLE001
            self._httpx = None

    def post_json(self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float) -> dict[str, Any]:  # noqa: A003
        if self._httpx is None:
            raise RuntimeError("httpx is not installed")
        httpx = self._httpx
        with httpx.Client(timeout=timeout, headers=headers) as client:  # type: ignore[attr-defined]
            resp = client.post(url, json=json)
            resp.raise_for_status()
            return resp.json()


class Translator(Protocol):
    def translate(self, text: str, target_lang: str) -> str:
        ...


@dataclass
class RetryConfig:
    retries: int = 3
    backoff_base: float = 1.0
    timeout_seconds: float = 15.0


class TranslationError(Exception):
    pass


def _request_with_retries(fn: Callable[[], dict[str, Any]], *, retry: RetryConfig) -> dict[str, Any]:
    last_exc: Optional[Exception] = None
    for attempt in range(retry.retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "Translation request failed, retrying",
                extra={"attempt": attempt + 1, "retries": retry.retries, "error": str(exc)},
            )
            time.sleep(retry.backoff_base * (2**attempt))
    raise TranslationError(str(last_exc))


# --------------------------------- Adapters ------------------------------------


class DeepLTranslator:
    def __init__(self, api_key: str, requester: Optional[Requester] = None, retry: Optional[RetryConfig] = None) -> None:
        self.api_key = api_key
        self.requester = requester or HttpxRequester()
        self.retry = retry or RetryConfig()
        self.url = os.getenv("DEEPL_API_URL", "https://api-free.deepl.com/v2/translate")

    def translate(self, text: str, target_lang: str) -> str:  # type: ignore[override]
        headers = {"Authorization": f"DeepL-Auth-Key {self.api_key}"}
        payload = {"text": [text], "target_lang": target_lang.upper()}

        def call() -> dict[str, Any]:
            return self.requester.post_json(self.url, headers=headers, json=payload, timeout=self.retry.timeout_seconds)

        data = _request_with_retries(call, retry=self.retry)
        translations = data.get("translations") or []
        if not translations:
            raise TranslationError("No translations in response")
        return str(translations[0].get("text") or "")


class GoogleTranslator:
    def __init__(self, api_key: str, requester: Optional[Requester] = None, retry: Optional[RetryConfig] = None) -> None:
        self.api_key = api_key
        self.requester = requester or HttpxRequester()
        self.retry = retry or RetryConfig()
        self.url = os.getenv(
            "GOOGLE_TRANSLATE_API_URL",
            "https://translation.googleapis.com/language/translate/v2",
        )

    def translate(self, text: str, target_lang: str) -> str:  # type: ignore[override]
        headers: dict[str, str] = {}
        payload = {"q": text, "target": target_lang, "key": self.api_key}

        def call() -> dict[str, Any]:
            return self.requester.post_json(self.url, headers=headers, json=payload, timeout=self.retry.timeout_seconds)

        data = _request_with_retries(call, retry=self.retry)
        translations = ((data.get("data") or {}).get("translations")) or []
        if not translations:
            raise TranslationError("No translations in response")
        return str(translations[0].get("translatedText") or "")


class YandexTranslator:
    def __init__(self, api_key: str, folder_id: Optional[str] = None, requester: Optional[Requester] = None, retry: Optional[RetryConfig] = None) -> None:
        self.api_key = api_key
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID", "")
        self.requester = requester or HttpxRequester()
        self.retry = retry or RetryConfig()
        self.url = os.getenv(
            "YANDEX_TRANSLATE_API_URL",
            "https://translate.api.cloud.yandex.net/translate/v2/translate",
        )

    def translate(self, text: str, target_lang: str) -> str:  # type: ignore[override]
        headers = {"Authorization": f"Api-Key {self.api_key}"}
        payload = {"texts": [text], "targetLanguageCode": target_lang, "folderId": self.folder_id or None}

        def call() -> dict[str, Any]:
            return self.requester.post_json(self.url, headers=headers, json=payload, timeout=self.retry.timeout_seconds)

        data = _request_with_retries(call, retry=self.retry)
        translations = data.get("translations") or []
        if not translations:
            raise TranslationError("No translations in response")
        return str(translations[0].get("text") or "")


def get_translator(provider: str, *, requester: Optional[Requester] = None) -> Translator:
    provider_norm = (provider or "").strip().lower()
    retry = RetryConfig(
        retries=int(os.getenv("TRANSLATE_RETRIES", "3")),
        backoff_base=float(os.getenv("TRANSLATE_BACKOFF", "1")),
        timeout_seconds=float(os.getenv("TRANSLATE_TIMEOUT", "15")),
    )
    if provider_norm in {"deepl", "deep_l"}:
        api_key = os.getenv("DEEPL_API_KEY", "test-key")
        return DeepLTranslator(api_key=api_key, requester=requester, retry=retry)
    if provider_norm in {"google", "gcp"}:
        api_key = os.getenv("GOOGLE_API_KEY", "test-key")
        return GoogleTranslator(api_key=api_key, requester=requester, retry=retry)
    if provider_norm in {"yandex", "yc"}:
        api_key = os.getenv("YANDEX_API_KEY", "test-key")
        folder = os.getenv("YANDEX_FOLDER_ID")
        return YandexTranslator(api_key=api_key, folder_id=folder, requester=requester, retry=retry)
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def _find_or_create_translation(session: Session, *, article_id: int, lang: str, provider: str, text: str) -> int:
    # Idempotent lookup
    existing = session.execute(
        select(Translation).where(
            Translation.article_id == article_id, Translation.lang == lang, Translation.provider == provider
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing.id
    tr = Translation(article_id=article_id, lang=lang, provider=provider, text=text)
    session.add(tr)
    session.flush()
    return tr.id


def translate_article_core(
    session: Session,
    *,
    article_id: int,
    lang_target: str,
    provider: Optional[str] = None,
    requester: Optional[Requester] = None,
) -> int:
    """Translate an article and persist translation. Returns translation_id.

    - Enforces max input length via TRANSLATE_MAX_CHARS (default 4000)
    - Uses provider from argument or LLM_PROVIDER env var
    - Idempotent on (article_id, lang, provider)
    """
    provider_name = provider or os.getenv("LLM_PROVIDER", "deepl")
    translator = get_translator(provider_name, requester=requester)

    article = session.get(Article, article_id)
    if article is None:
        raise ValueError(f"Article {article_id} not found")

    max_chars = int(os.getenv("TRANSLATE_MAX_CHARS", "4000"))
    text = (article.content or "").strip()
    if len(text) > max_chars:
        logger.info("Truncating article content for translation", extra={"article_id": article_id, "max_chars": max_chars})
        text = text[:max_chars]

    translated = translator.translate(text, lang_target)
    tr_id = _find_or_create_translation(
        session, article_id=article_id, lang=lang_target, provider=provider_name.lower(), text=translated
    )
    session.commit()
    logger.info(
        "Article translated",
        extra={"article_id": article_id, "lang": lang_target, "provider": provider_name.lower(), "translation_id": tr_id},
    )
    return tr_id


# --------------------------------- Celery task ---------------------------------


def _task_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    try:
        from celery import shared_task  # type: ignore

        return shared_task(name="app.tasks.translate_article")(func)  # type: ignore[return-value]
    except Exception:
        return func


@_task_decorator
def translate_article(article_id: int, lang_target: str) -> int:
    """Celery-задача: перевести статью и сохранить перевод."""
    session = create_session_from_env()
    try:
        return translate_article_core(session, article_id=article_id, lang_target=lang_target)
    finally:
        session.close()


__all__ = [
    "Translation",
    "Translator",
    "DeepLTranslator",
    "GoogleTranslator",
    "YandexTranslator",
    "translate_article",
    "translate_article_core",
]
