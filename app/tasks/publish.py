"""
Publishing tasks:

- Scheduler `schedule_posts()` distributes approved content into `schedules` by cron,
  respecting per-day limits and schedule timezones.
- Executor `publish_post(post_id)` publishes to Telegram or VK and updates status/logs.

Designed to be self-contained and testable; Celery decorators are optional.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

from sqlalchemy import JSON as SAJSON
from sqlalchemy import DateTime, Enum as SAEnum, ForeignKey, Integer, String, and_, func, select
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship

from app.tasks.ingest import Base, create_session_from_env
from app.tasks.moderate import Moderation, ModerationDecision, ModerationTarget
from app.tasks.rewrite import Rewrite
from app.tasks.translate import Requester, HttpxRequester

logger = logging.getLogger("app.tasks.publish")


class Platform(str, Enum):
    TELEGRAM = "telegram"
    VK = "vk"


class PostStatus(str, Enum):
    APPROVED = "approved"  # ready for scheduling
    SCHEDULED = "scheduled"
    PUBLISHED = "published"
    FAILED = "failed"


class Schedule(Base):
    __tablename__ = "schedules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    platform: Mapped[Platform] = mapped_column(SAEnum(Platform), nullable=False)
    cron: Mapped[str] = mapped_column(String(64), nullable=False)  # supports patterns like "*/10 * * * *" or "m h * * *"
    timezone: Mapped[str] = mapped_column(String(64), nullable=False, default="UTC")
    daily_limit: Mapped[int] = mapped_column(Integer, nullable=False, default=24)

    posts: Mapped[list[Post]] = relationship("Post", back_populates="schedule")  # type: ignore[name-defined]


class Post(Base):
    __tablename__ = "posts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    schedule_id: Mapped[int] = mapped_column(ForeignKey("schedules.id"), nullable=False, index=True)
    rewrite_id: Mapped[int] = mapped_column(ForeignKey("rewrites.id"), nullable=False, index=True)
    platform: Mapped[Platform] = mapped_column(SAEnum(Platform), nullable=False)
    status: Mapped[PostStatus] = mapped_column(SAEnum(PostStatus), nullable=False, default=PostStatus.SCHEDULED)
    text: Mapped[str] = mapped_column(String, nullable=False)
    media_url: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    platform_post_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    log: Mapped[Optional[dict[str, Any]]] = mapped_column(SAJSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    schedule: Mapped[Schedule] = relationship(Schedule, back_populates="posts")
    rewrite: Mapped[Rewrite] = relationship(Rewrite, lazy="joined")


# ----------------------------- Cron helper (simple) -----------------------------


def _parse_cron_min_hour(expr: str) -> tuple[str, str]:
    parts = expr.strip().split()
    if len(parts) < 2:
        # default: every minute
        return "*", "*"
    return parts[0], parts[1]


def _next_run_from(expr: str, now_local: datetime) -> datetime:
    """Compute next run time for a very small subset of cron:

    Supported patterns for minutes/hours:
    - "*" every minute/hour
    - "*/N" step
    - exact number (0-59 or 0-23)
    We ignore day-of-month/day-of-week fields.
    """
    m_expr, h_expr = _parse_cron_min_hour(expr)

    def minutes_for(hour: int) -> list[int]:
        if m_expr == "*":
            return list(range(60))
        if m_expr.startswith("*/"):
            step = max(1, int(m_expr[2:]))
            return list(range(0, 60, step))
        return [int(m_expr)]

    def hours() -> list[int]:
        if h_expr == "*":
            return list(range(24))
        if h_expr.startswith("*/"):
            step = max(1, int(h_expr[2:]))
            return list(range(0, 24, step))
        return [int(h_expr)]

    candidate = now_local.replace(second=0, microsecond=0) + timedelta(minutes=1)
    # search up to 7 days ahead
    for _ in range(0, 60 * 24 * 7):
        if candidate.hour in set(hours()) and candidate.minute in set(minutes_for(candidate.hour)):
            return candidate
        candidate += timedelta(minutes=1)
    return candidate


# ------------------------------- Scheduling logic ------------------------------


def _find_approved_rewrites(session: Session) -> list[Rewrite]:
    # Rewrites with moderation OK and not yet scheduled/published in posts
    # Select latest rewrite rows that don't have an associated Post in non-failed state
    subq = select(Post.rewrite_id).where(Post.status.in_([PostStatus.SCHEDULED, PostStatus.PUBLISHED])).subquery()
    q = (
        select(Rewrite)
        .join(Moderation, and_(Moderation.target_type == ModerationTarget.REWRITE, Moderation.target_id == Rewrite.id))
        .where(Moderation.decision == ModerationDecision.OK)
        .where(~Rewrite.id.in_(select(subq.c.rewrite_id)))
    )
    return list(session.execute(q).scalars().all())


def schedule_posts_core(session: Session) -> list[int]:
    """Plan размещения одобренных материалов по расписаниям.

    Ищет `Rewrite` с решением модерации OK и создаёт `Post` в статусе
    `scheduled` согласно правилам `Schedule` (cron, timezone, daily_limit).
    Возвращает список созданных `post_id`. Идемпотентно: не создаёт дубликаты
    для уже запланированных/опубликованных переписанных материалов.
    """
    schedules = session.execute(select(Schedule)).scalars().all()
    if not schedules:
        return []
    rewrites = _find_approved_rewrites(session)
    new_post_ids: list[int] = []
    for sched in schedules:
        tz = ZoneInfo(sched.timezone)
        now_local = datetime.now(tz)
        # Count how many already scheduled today (local date)
        start_day = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        end_day = start_day + timedelta(days=1)
        start_utc = start_day.astimezone(timezone.utc)
        end_utc = end_day.astimezone(timezone.utc)
        count_today = session.execute(
            select(func.count(Post.id)).where(
                Post.schedule_id == sched.id,
                Post.status == PostStatus.SCHEDULED,
                Post.scheduled_at >= start_utc,
                Post.scheduled_at < end_utc,
            )
        ).scalar_one()
        remaining = max(0, sched.daily_limit - int(count_today))
        if remaining <= 0:
            continue

        # Allocate posts to next available slots
        next_time_local = _next_run_from(sched.cron, now_local)
        for _ in range(min(remaining, len(rewrites))):
            if not rewrites:
                break
            rw = rewrites.pop(0)
            scheduled_at = next_time_local.astimezone(timezone.utc)
            post = Post(
                schedule_id=sched.id,
                rewrite_id=rw.id,
                platform=sched.platform,
                status=PostStatus.SCHEDULED,
                text=rw.text,
                media_url=None,
                scheduled_at=scheduled_at,
                log=None,
            )
            session.add(post)
            session.flush()
            new_post_ids.append(post.id)
            # advance to next slot after this one
            next_time_local = _next_run_from(sched.cron, next_time_local)

        session.commit()
    logger.info("Scheduling completed", extra={"created": len(new_post_ids)})
    return new_post_ids


# -------------------------------- Publish logic --------------------------------


def _publish_telegram(post: Post, *, requester: Requester) -> tuple[bool, str, dict[str, Any]]:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "test-token")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "-1001234567890")
    base = f"https://api.telegram.org/bot{token}"
    if post.media_url:
        url = f"{base}/sendPhoto"
        payload = {"chat_id": chat_id, "photo": post.media_url, "caption": post.text}
    else:
        url = f"{base}/sendMessage"
        payload = {"chat_id": chat_id, "text": post.text}
    data = requester.post_json(url, headers={}, json=payload, timeout=15.0)
    ok = bool(data.get("ok", True))
    msg = data.get("result") or {}
    msg_id = str((msg.get("message_id") or "0"))
    return ok, msg_id, data


def _publish_vk(post: Post, *, requester: Requester) -> tuple[bool, str, dict[str, Any]]:
    token = os.getenv("VK_ACCESS_TOKEN", "test-token")
    group_id = int(os.getenv("VK_GROUP_ID", "123456"))
    api_url = "https://api.vk.com/method/wall.post"
    payload = {
        "owner_id": -abs(group_id),
        "from_group": 1,
        "message": post.text,
        "access_token": token,
        "v": os.getenv("VK_API_VERSION", "5.199"),
    }
    if post.media_url:
        # For tests, we just pass URL as attachment; real flow would upload and save photo first.
        payload["attachments"] = post.media_url
    data = requester.post_json(api_url, headers={}, json=payload, timeout=15.0)
    resp = data.get("response") or {}
    post_id = str(resp.get("post_id") or "0")
    ok = bool(post_id and post_id != "0")
    return ok, post_id, data


def publish_post_core(session: Session, *, post_id: int, requester: Optional[Requester] = None) -> int:
    """Опубликовать один пост в целевую платформу.

    По `post_id` выбирает `Post`, вызывает соответствующий публикатор (Telegram/VK),
    сохраняет `platform_post_id`, лог ответа и выставляет статус `published` или `failed`.
    Возвращает `post_id`.
    """
    post = session.get(Post, post_id)
    if post is None:
        raise ValueError(f"Post {post_id} not found")

    req = requester or HttpxRequester()
    try:
        if post.platform == Platform.TELEGRAM:
            ok, ext_id, raw = _publish_telegram(post, requester=req)
        elif post.platform == Platform.VK:
            ok, ext_id, raw = _publish_vk(post, requester=req)
        else:
            raise ValueError(f"Unsupported platform: {post.platform}")

        post.platform_post_id = ext_id
        post.log = {"response": raw}
        post.status = PostStatus.PUBLISHED if ok else PostStatus.FAILED
        session.add(post)
        session.commit()
    except Exception as exc:  # noqa: BLE001
        post.status = PostStatus.FAILED
        post.log = {"error": str(exc)}
        session.add(post)
        session.commit()
        logger.exception("Failed to publish post", extra={"post_id": post_id})
        raise
    return post.id


# --------------------------------- Celery tasks --------------------------------


def _task_decorator(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        try:
            from celery import shared_task  # type: ignore

            return shared_task(name=name)(func)  # type: ignore[return-value]
        except Exception:
            return func

    return wrap


@_task_decorator("app.tasks.schedule_posts")
def schedule_posts() -> list[int]:
    """Celery-задача: планирование постов по расписанию."""
    session = create_session_from_env()
    try:
        return schedule_posts_core(session)
    finally:
        session.close()


@_task_decorator("app.tasks.publish_post")
def publish_post(post_id: int) -> int:
    """Celery-задача: публикация одного поста."""
    session = create_session_from_env()
    try:
        return publish_post_core(session, post_id=post_id)
    finally:
        session.close()


__all__ = [
    "Schedule",
    "Post",
    "Platform",
    "PostStatus",
    "schedule_posts",
    "schedule_posts_core",
    "publish_post",
    "publish_post_core",
]
