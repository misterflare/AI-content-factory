"""
Rewrite task that adapts translated text for a specific client.

- Celery task: rewrite_for_client(translation_id: int, client_id: int)
- Fetches client brief, tone, required_hashtags, forbidden_words
- Constructs concise prompt (300–600 chars, 2–4 hashtags, avoid forbidden words, add 1 CTA)
- Calls LLM via OpenAI-compatible interface with retries and token truncation
- Persists result into rewrites table; idempotent per (translation_id, client_id)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Protocol

from sqlalchemy import JSON as SAJSON
from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint, select
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship

from app.tasks.ingest import Base, create_session_from_env
from app.tasks.translate import Translation

logger = logging.getLogger("app.tasks.rewrite")


# ------------------------------------ ORM --------------------------------------


class Client(Base):
    __tablename__ = "clients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    brief: Mapped[str] = mapped_column(String, nullable=False)
    tone: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    required_hashtags: Mapped[Optional[list[str]]] = mapped_column(SAJSON, nullable=True)
    forbidden_words: Mapped[Optional[list[str]]] = mapped_column(SAJSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )


class Rewrite(Base):
    __tablename__ = "rewrites"
    __table_args__ = (
        UniqueConstraint("translation_id", "client_id", name="uq_rewrite_translation_client"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    translation_id: Mapped[int] = mapped_column(ForeignKey("translations.id"), nullable=False, index=True)
    client_id: Mapped[int] = mapped_column(ForeignKey("clients.id"), nullable=False, index=True)
    text: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    translation: Mapped[Translation] = relationship(Translation, lazy="joined")
    client: Mapped[Client] = relationship(Client, lazy="joined")


# ------------------------------- LLM Abstraction --------------------------------


class Requester(Protocol):
    def post_json(self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float) -> dict[str, Any]:
        ...


class HttpxRequester:
    def __init__(self) -> None:
        try:
            import httpx as _httpx  # type: ignore

            self._httpx = _httpx
        except Exception:  # noqa: BLE001
            self._httpx = None

    def post_json(self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float) -> dict[str, Any]:  # type: ignore[override]
        if self._httpx is None:
            raise RuntimeError("httpx is not installed")
        httpx = self._httpx
        with httpx.Client(timeout=timeout, headers=headers) as client:  # type: ignore[attr-defined]
            resp = client.post(url, json=json)
            resp.raise_for_status()
            return resp.json()


class LLMClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str, *, max_tokens: int) -> str:
        ...


@dataclass
class RetryConfig:
    retries: int = 3
    backoff_base: float = 1.0
    timeout_seconds: float = 15.0


class LLMError(Exception):
    pass


def _with_retries(fn: Callable[[], dict[str, Any]], *, retry: RetryConfig) -> dict[str, Any]:
    last_exc: Optional[Exception] = None
    for attempt in range(retry.retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "LLM request failed, retrying",
                extra={"attempt": attempt + 1, "retries": retry.retries, "error": str(exc)},
            )
            time.sleep(retry.backoff_base * (2**attempt))
    raise LLMError(str(last_exc))


def _approx_token_truncate(text: str, *, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    # Approximate tokens by whitespace-split pieces limited to max_tokens.
    parts = re.findall(r"\S+", text)
    if len(parts) <= max_tokens:
        return text
    return " ".join(parts[:max_tokens])


class OpenAIChatClient:
    """OpenAI-compatible Chat Completions client."""

    def __init__(self, *, api_key: str, base_url: Optional[str] = None, model: Optional[str] = None, requester: Optional[Requester] = None, retry: Optional[RetryConfig] = None) -> None:
        self.api_key = api_key
        self.base_url = (base_url or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.requester = requester or HttpxRequester()
        self.retry = retry or RetryConfig()

    def generate(self, system_prompt: str, user_prompt: str, *, max_tokens: int) -> str:  # type: ignore[override]
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.5,
            "max_tokens": max_tokens,
        }

        def call() -> dict[str, Any]:
            return self.requester.post_json(url, headers=headers, json=payload, timeout=self.retry.timeout_seconds)

        data = _with_retries(call, retry=self.retry)
        choices = data.get("choices") or []
        if not choices:
            raise LLMError("No choices in response")
        msg = (choices[0].get("message") or {})
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMError("Empty content from LLM")
        return content.strip()


def get_llm_client(provider: Optional[str] = None, *, requester: Optional[Requester] = None) -> LLMClient:
    prov = (provider or os.getenv("LLM_PROVIDER", "openai")).strip().lower()
    retry = RetryConfig(
        retries=int(os.getenv("LLM_RETRIES", os.getenv("TRANSLATE_RETRIES", "3"))),
        backoff_base=float(os.getenv("LLM_BACKOFF", os.getenv("TRANSLATE_BACKOFF", "1"))),
        timeout_seconds=float(os.getenv("LLM_TIMEOUT", os.getenv("TRANSLATE_TIMEOUT", "15"))),
    )
    if prov in {"openai", "oai"}:
        key = os.getenv("OPENAI_API_KEY", "test-key")
        base = os.getenv("OPENAI_API_BASE")
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        return OpenAIChatClient(api_key=key, base_url=base, model=model, requester=requester, retry=retry)
    # For other OpenAI-compatible providers, reuse OpenAIChatClient with custom base/model
    key = os.getenv("OPENAI_API_KEY", "test-key")
    base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    return OpenAIChatClient(api_key=key, base_url=base, model=model, requester=requester, retry=retry)


# --------------------------------- Core logic ----------------------------------


def _build_prompt(*, brief: str, tone: Optional[str], required_hashtags: Optional[list[str]], forbidden_words: Optional[list[str]], source_text: str) -> tuple[str, str, int]:
    system = "Вы — опытный редактор контента для соцсетей. Пишите кратко, ясно, без воды."
    req_tags = ", ".join(required_hashtags or [])
    forb = ", ".join(forbidden_words or [])
    tone_desc = f"Тон: {tone}." if tone else ""
    instructions = (
        "Перепиши кратко (300–600 знаков), добавь 2–4 хэштега из списка, "
        "избегай запрещённых слов, добавь 1 призыв к действию (CTA)."
    )
    user = (
        f"Бриф клиента: {brief}\n"
        f"{tone_desc}\n"
        f"Список допустимых хэштегов: {req_tags}\n"
        f"Запрещённые слова: {forb}\n"
        f"Текст для переписывания:\n{source_text}"
    )
    # Allocate roughly for 600 chars output; approximate tokens ~= chars/4
    max_completion_tokens = int(os.getenv("REWRITE_MAX_TOKENS", "512"))
    return system, f"{instructions}\n\n{user}", max_completion_tokens


def _find_or_create_rewrite(session: Session, *, translation_id: int, client_id: int, text: str) -> int:
    existing = session.execute(
        select(Rewrite).where(Rewrite.translation_id == translation_id, Rewrite.client_id == client_id)
    ).scalar_one_or_none()
    if existing is not None:
        return existing.id
    rw = Rewrite(translation_id=translation_id, client_id=client_id, text=text)
    session.add(rw)
    session.flush()
    return rw.id


def rewrite_for_client_core(
    session: Session,
    *,
    translation_id: int,
    client_id: int,
    provider: Optional[str] = None,
    requester: Optional[Requester] = None,
) -> int:
    tr = session.get(Translation, translation_id)
    if tr is None:
        raise ValueError(f"Translation {translation_id} not found")
    cl = session.get(Client, client_id)
    if cl is None:
        raise ValueError(f"Client {client_id} not found")

    llm = get_llm_client(provider, requester=requester)

    # Truncate input tokens approx to keep prompt within provider limits
    max_input_tokens = int(os.getenv("REWRITE_MAX_INPUT_TOKENS", "4000"))
    source_text = _approx_token_truncate(tr.text, max_tokens=max_input_tokens)

    system, user, max_completion = _build_prompt(
        brief=cl.brief,
        tone=cl.tone,
        required_hashtags=cl.required_hashtags,
        forbidden_words=cl.forbidden_words,
        source_text=source_text,
    )

    out = llm.generate(system, user, max_tokens=max_completion)
    rw_id = _find_or_create_rewrite(session, translation_id=translation_id, client_id=client_id, text=out)
    session.commit()
    logger.info(
        "Rewrite created",
        extra={"translation_id": translation_id, "client_id": client_id, "rewrite_id": rw_id},
    )
    return rw_id


# --------------------------------- Celery task ---------------------------------


def _task_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    try:
        from celery import shared_task  # type: ignore

        return shared_task(name="app.tasks.rewrite_for_client")(func)  # type: ignore[return-value]
    except Exception:
        return func


@_task_decorator
def rewrite_for_client(translation_id: int, client_id: int) -> int:
    session = create_session_from_env()
    try:
        return rewrite_for_client_core(session, translation_id=translation_id, client_id=client_id)
    finally:
        session.close()


__all__ = [
    "Client",
    "Rewrite",
    "LLMClient",
    "OpenAIChatClient",
    "rewrite_for_client",
    "rewrite_for_client_core",
]

