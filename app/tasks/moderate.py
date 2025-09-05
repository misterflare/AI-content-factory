"""
Moderation task for rewrites (and optionally articles).

Rules:
- stop-words, topics (allowed/blocked), length, presence of source
- LLM classification: ok / needs_review / reject (OpenAI-compatible client)
- Persists result in `moderation` table with reasons[] (JSON array)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Iterable, Optional

from sqlalchemy import JSON as SAJSON
from sqlalchemy import DateTime, Enum as SAEnum, ForeignKey, Integer, String, UniqueConstraint, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from app.tasks.ingest import Article, Base, create_session_from_env
from app.tasks.rewrite import Rewrite, get_llm_client, LLMClient

logger = logging.getLogger("app.tasks.moderate")


class ModerationTarget(str, Enum):
    REWRITE = "rewrite"
    ARTICLE = "article"


class ModerationDecision(str, Enum):
    OK = "ok"
    NEEDS_REVIEW = "needs_review"
    REJECT = "reject"


class Moderation(Base):
    __tablename__ = "moderation"
    __table_args__ = (
        UniqueConstraint("target_type", "target_id", name="uq_moderation_target"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    target_type: Mapped[ModerationTarget] = mapped_column(SAEnum(ModerationTarget), nullable=False)
    target_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    decision: Mapped[ModerationDecision] = mapped_column(SAEnum(ModerationDecision), nullable=False)
    reasons: Mapped[list[str]] = mapped_column(SAJSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)


# --------------------------------- Helpers -------------------------------------


def _split_csv_env(name: str) -> list[str]:
    val = os.getenv(name, "").strip()
    if not val:
        return []
    return [s.strip() for s in val.split(",") if s.strip()]


def _contains_url(text: str) -> bool:
    return bool(re.search(r"https?://[\w\-._~:/?#\[\]@!$&'()*+,;=%]+", text))


def _contains_any(text: str, words: Iterable[str]) -> Optional[str]:
    text_low = text.lower()
    for w in words:
        w_clean = w.strip().lower()
        if not w_clean:
            continue
        if w_clean in text_low:
            return w
    return None


def _match_any(text: str, keywords: Iterable[str]) -> Optional[str]:
    # simple substring match for topics/keywords
    return _contains_any(text, keywords)


def _trim_chars(text: str) -> str:
    return text.strip()


def _classify_with_llm(text: str, *, llm: LLMClient, rules_summary: str, max_input_tokens: int = 4000) -> ModerationDecision:
    # We instruct LLM to output strictly one token among defined classes.
    system = (
        "Вы — автоматический модератор контента. Возвращайте только одно слово: "
        "ok, needs_review или reject — без пояснений."
    )
    user = (
        "Оцените текст по правилам. Правила: "
        f"{rules_summary}\n\n"
        f"Текст:\n{text}"
    )
    # Keep the response tiny
    out = llm.generate(system, user, max_tokens=4).strip().lower()
    if "reject" in out:
        return ModerationDecision.REJECT
    if "needs_review" in out or "needs-review" in out or "review" in out:
        return ModerationDecision.NEEDS_REVIEW
    return ModerationDecision.OK


# ---------------------------------- Core ---------------------------------------


@dataclass
class RuleConfig:
    stop_words: list[str]
    allowed_topics: list[str]
    blocked_topics: list[str]
    min_chars: int
    max_chars: int
    require_source: bool


def _load_rule_config() -> RuleConfig:
    return RuleConfig(
        stop_words=_split_csv_env("MODERATION_STOP_WORDS"),
        allowed_topics=_split_csv_env("MODERATION_ALLOWED_TOPICS"),
        blocked_topics=_split_csv_env("MODERATION_BLOCKED_TOPICS"),
        min_chars=int(os.getenv("MODERATION_MIN_CHARS", os.getenv("REWRITE_MIN_CHARS", "300"))),
        max_chars=int(os.getenv("MODERATION_MAX_CHARS", os.getenv("REWRITE_MAX_CHARS", "600"))),
        require_source=os.getenv("MODERATION_REQUIRE_SOURCE", "1") not in {"0", "false", "False"},
    )


def _find_or_create_moderation(session: Session, *, target_type: ModerationTarget, target_id: int, decision: ModerationDecision, reasons: list[str]) -> int:
    existing = session.execute(
        select(Moderation).where(Moderation.target_type == target_type, Moderation.target_id == target_id)
    ).scalar_one_or_none()
    if existing is not None:
        return existing.id
    row = Moderation(target_type=target_type, target_id=target_id, decision=decision, reasons=reasons)
    session.add(row)
    session.flush()
    return row.id


def moderate_rewrite_core(
    session: Session,
    *,
    rewrite_id: int,
    llm: Optional[LLMClient] = None,
    rules: Optional[RuleConfig] = None,
) -> int:
    """Модерация переписанного текста по правилам и LLM.

    Возвращает `moderation_id`. Идемпотентно: при повторном вызове для того же
    `rewrite_id` вернёт уже существующую запись.
    """
    rw = session.get(Rewrite, rewrite_id)
    if rw is None:
        raise ValueError(f"Rewrite {rewrite_id} not found")
    tr = rw.translation
    art = tr.article

    rules = rules or _load_rule_config()
    text = _trim_chars(rw.text)
    reasons: list[str] = []

    # Length checks
    n = len(text)
    if n < rules.min_chars:
        reasons.append("too_short")
    if n > rules.max_chars:
        reasons.append("too_long")

    # Stop-words
    sw = _contains_any(text, rules.stop_words)
    if sw:
        reasons.append(f"stop_word:{sw}")

    # Topics
    blocked = _match_any(text, rules.blocked_topics)
    if blocked:
        reasons.append(f"blocked_topic:{blocked}")
    if rules.allowed_topics:
        allowed = _match_any(text, rules.allowed_topics)
        if not allowed:
            reasons.append("no_allowed_topic")

    # Source presence
    if rules.require_source:
        if not _contains_url(text):
            # also accept presence of original article URL domain
            if not (art and art.url and art.url in text):
                reasons.append("no_source")

    # LLM classification
    provider = os.getenv("LLM_PROVIDER", "openai")
    client = llm or get_llm_client(provider)
    rule_summary = (
        f"min_chars={rules.min_chars}, max_chars={rules.max_chars}, "
        f"stop_words={rules.stop_words}, allowed_topics={rules.allowed_topics}, blocked_topics={rules.blocked_topics}, "
        f"require_source={rules.require_source}"
    )
    llm_decision = _classify_with_llm(text, llm=client, rules_summary=rule_summary)

    # Merge decision
    if llm_decision == ModerationDecision.REJECT:
        decision = ModerationDecision.REJECT
    elif reasons or llm_decision == ModerationDecision.NEEDS_REVIEW:
        decision = ModerationDecision.NEEDS_REVIEW
    else:
        decision = ModerationDecision.OK

    mod_id = _find_or_create_moderation(
        session, target_type=ModerationTarget.REWRITE, target_id=rewrite_id, decision=decision, reasons=reasons
    )
    session.commit()
    logger.info(
        "Moderation stored",
        extra={"rewrite_id": rewrite_id, "decision": decision.value, "reasons": reasons, "id": mod_id},
    )
    return mod_id


# --------------------------------- Celery task ---------------------------------


def _task_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    try:
        from celery import shared_task  # type: ignore

        return shared_task(name="app.tasks.moderate_rewrite")(func)  # type: ignore[return-value]
    except Exception:
        return func


@_task_decorator
def moderate_rewrite(rewrite_id: int) -> int:
    """Celery-задача: модерация `Rewrite` по идентификатору."""
    session = create_session_from_env()
    try:
        return moderate_rewrite_core(session, rewrite_id=rewrite_id)
    finally:
        session.close()


__all__ = [
    "Moderation",
    "ModerationDecision",
    "ModerationTarget",
    "moderate_rewrite",
    "moderate_rewrite_core",
]
