"""
Ingestion task for pulling new articles from configured sources.

- Supports RSS/Atom (feedparser if available, with XML fallback)
- Supports HTML via readability-lxml if available (simple fallback otherwise)
- Uses an HTTP client with timeout and retries (httpx if available)
- Computes content_hash (SHA256 of normalized text content)
- Persists only new articles unique by (source_id, content_hash)
- Returns list of new article IDs
- Logs counters: success, skipped, errors

Note: Celery, httpx, feedparser, readability are optional at import time to
keep tests runnable in constrained environments. The task decorates with a
no-op if Celery is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import logging
import os
import re
import time
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import Boolean, DateTime, Enum as SAEnum, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy import create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship
from sqlalchemy.types import JSON

logger = logging.getLogger("app.tasks.ingest")


class Base(DeclarativeBase):
    pass


class SourceType(str, Enum):
    RSS = "rss"
    ATOM = "atom"
    HTML = "html"


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    type: Mapped[SourceType] = mapped_column(SAEnum(SourceType), nullable=False)
    url: Mapped[str] = mapped_column(String(500), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    articles: Mapped[List[Article]] = relationship("Article", back_populates="source")  # type: ignore[name-defined]


class Article(Base):
    __tablename__ = "articles"
    __table_args__ = (
        UniqueConstraint("source_id", "content_hash", name="uq_article_source_content_hash"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(String, nullable=False)
    lead: Mapped[str] = mapped_column(String(500), nullable=False)
    authors: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    url: Mapped[str] = mapped_column(String(1000), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    source: Mapped[Source] = relationship("Source", back_populates="articles")  # type: ignore[name-defined]


# --------------------------- HTTP client with retries ---------------------------


class HttpError(Exception):
    """HTTP retrieval error."""


@dataclass
class HttpResponse:
    url: str
    status_code: int
    text: str
    headers: dict[str, str]


class HttpClient:
    """Simple HTTP client with timeout and retries using httpx if available."""

    def __init__(self, timeout_seconds: float = 10.0, retries: int = 3, backoff_base: float = 1.0) -> None:
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.backoff_base = backoff_base

    def get_text(self, url: str, *, headers: Optional[dict[str, str]] = None) -> str:
        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt < self.retries:
            try:
                text = self._request_text(url, headers=headers)
                return text
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                sleep_for = self.backoff_base * (2 ** attempt)
                logger.warning("HTTP get failed, retrying", extra={"url": url, "attempt": attempt + 1, "retries": self.retries, "error": str(exc)})
                time.sleep(sleep_for)
                attempt += 1
        raise HttpError(f"Failed to GET {url}: {last_exc}")

    def _request_text(self, url: str, *, headers: Optional[dict[str, str]] = None) -> str:
        try:
            import httpx  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise HttpError("httpx is not installed") from exc

        with httpx.Client(timeout=self.timeout_seconds, headers=headers) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.text


def get_default_http_client() -> HttpClient:
    return HttpClient(timeout_seconds=10.0, retries=3, backoff_base=1.0)


# ----------------------------- Parsing and helpers -----------------------------


def _strip_html(html: str) -> str:
    # Very simple HTML tag stripper; good enough for normalization.
    # Remove script/style
    html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.IGNORECASE)
    # Remove tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _lead_from_text(text: str, max_len: int = 240) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _to_datetime_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_rss_or_atom(text: str) -> list[dict[str, Any]]:
    """Parse RSS/Atom content into a uniform list of entries.

    Tries feedparser first; if unavailable, uses a minimal XML fallback
    to support unit tests without external deps.
    """
    try:
        import feedparser  # type: ignore

        feed = feedparser.parse(text)
        entries: list[dict[str, Any]] = []
        for e in getattr(feed, "entries", []):
            title = getattr(e, "title", None) or (e.get("title") if isinstance(e, dict) else None)
            link = getattr(e, "link", None) or (e.get("link") if isinstance(e, dict) else None)
            content_html = None
            if hasattr(e, "content"):
                try:
                    content_html = e.content[0].value  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    pass
            summary = getattr(e, "summary", None) or (e.get("summary") if isinstance(e, dict) else None)
            authors: Optional[list[str]] = None
            if hasattr(e, "authors"):
                try:
                    authors = [a.get("name") for a in e.authors if isinstance(a, dict) and a.get("name")]  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    pass
            if not authors:
                author = getattr(e, "author", None) or (e.get("author") if isinstance(e, dict) else None)
                if isinstance(author, str):
                    authors = [author]

            dt: Optional[datetime] = None
            try:
                from datetime import datetime as _dt
                import time as _time

                st = getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
                if st is not None:
                    dt = _dt.fromtimestamp(_time.mktime(st))
            except Exception:  # noqa: BLE001
                dt = None

            entries.append(
                {
                    "title": str(title or "").strip(),
                    "url": str(link or "").strip(),
                    "content_html": str((content_html or summary or "")).strip(),
                    "authors": authors or None,
                    "published_at": _to_datetime_utc(dt),
                }
            )
        return entries
    except Exception:  # noqa: BLE001
        # Minimal RSS/Atom fallback
        from xml.etree import ElementTree as ET

        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            return []

        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "dc": "http://purl.org/dc/elements/1.1/",
        }
        entries: list[dict[str, Any]] = []
        # RSS items
        for item in root.findall(".//item"):
            title = item.findtext("title") or ""
            link = item.findtext("link") or ""
            desc = item.findtext("description") or ""
            pub = item.findtext("pubDate")
            dt = None
            if pub:
                try:
                    from email.utils import parsedate_to_datetime

                    dt = parsedate_to_datetime(pub)
                except Exception:  # noqa: BLE001
                    dt = None
            entries.append({"title": title, "url": link, "content_html": desc, "authors": None, "published_at": _to_datetime_utc(dt)})

        # Atom entries
        for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
            title = entry.findtext("{http://www.w3.org/2005/Atom}title") or ""
            link_el = entry.find("{http://www.w3.org/2005/Atom}link")
            href = link_el.get("href") if link_el is not None else ""
            content = entry.findtext("{http://www.w3.org/2005/Atom}content") or entry.findtext("{http://www.w3.org/2005/Atom}summary") or ""
            updated = entry.findtext("{http://www.w3.org/2005/Atom}updated")
            dt = None
            if updated:
                try:
                    dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                except Exception:  # noqa: BLE001
                    dt = None
            entries.append({"title": title, "url": href, "content_html": content, "authors": None, "published_at": _to_datetime_utc(dt)})

        return entries


def parse_html_with_readability(html: str) -> tuple[str, str]:
    """Return (title, content_text) from HTML using readability if available.

    Falls back to very basic extraction if readability-lxml is not installed.
    """
    try:
        from readability import Document  # type: ignore

        doc = Document(html)
        title = doc.short_title() or doc.title() or ""
        summary_html = doc.summary() or ""
        content_text = _strip_html(summary_html)
        if not content_text:
            # fallback to full text
            content_text = _strip_html(html)
        return title.strip(), content_text.strip()
    except Exception:
        # Very naive fallback
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""
        content_text = _strip_html(html)
        return title, content_text


def normalize_article(fields: dict[str, Any]) -> dict[str, Any]:
    title = str(fields.get("title") or "").strip() or "Untitled"
    url = str(fields.get("url") or "").strip() or ""
    authors_raw = fields.get("authors")
    authors: Optional[list[str]]
    if isinstance(authors_raw, list):
        authors = [str(a).strip() for a in authors_raw if str(a).strip()]
    else:
        authors = None
    published_at = fields.get("published_at")
    if isinstance(published_at, str):
        try:
            published_at_dt = datetime.fromisoformat(published_at)
        except Exception:  # noqa: BLE001
            published_at_dt = None
    elif isinstance(published_at, datetime):
        published_at_dt = _to_datetime_utc(published_at)
    else:
        published_at_dt = None

    content_html = str(fields.get("content_html") or fields.get("content") or "")
    content_text = _strip_html(content_html)
    lead = _lead_from_text(content_text)
    content_hash = _sha256(content_text)

    return {
        "title": title,
        "content": content_text,
        "lead": lead,
        "authors": authors,
        "published_at": published_at_dt,
        "url": url,
        "content_hash": content_hash,
    }


# ------------------------------- DB functionality ------------------------------


def create_session_from_env() -> Session:
    url = os.getenv("DATABASE_URL", "sqlite:///./app_ingest.sqlite")
    engine = create_engine(url, future=True)
    Base.metadata.create_all(engine)
    return Session(engine, future=True)


def _get_active_sources(session: Session) -> list[Source]:
    stmt = select(Source).where(Source.active.is_(True))
    return list(session.execute(stmt).scalars().all())


def _save_article_if_new(session: Session, source: Source, data: dict[str, Any]) -> Optional[int]:
    # Fast check for existence
    stmt = select(Article.id).where(Article.source_id == source.id, Article.content_hash == data["content_hash"])  # type: ignore[index]
    existing = session.execute(stmt).scalar_one_or_none()
    if existing is not None:
        return None
    art = Article(
        source_id=source.id,
        title=data["title"],
        content=data["content"],
        lead=data["lead"],
        authors=data.get("authors"),
        published_at=data.get("published_at"),
        url=data["url"],
        content_hash=data["content_hash"],
    )
    session.add(art)
    session.flush()
    return art.id


# ------------------------------- Ingest pipeline -------------------------------


def ingest_sources(session: Session, *, http_client: Optional[HttpClient] = None) -> list[int]:
    """Ingest active sources. Returns IDs of newly created articles.

    This function contains the core logic and is separate from the Celery task
    wrapper for testability and idempotence.
    """
    client = http_client or get_default_http_client()
    sources = _get_active_sources(session)
    new_ids: list[int] = []
    success = 0
    skipped = 0
    errors = 0
    for src in sources:
        try:
            if src.type in (SourceType.RSS, SourceType.ATOM):
                text = client.get_text(src.url)
                entries = parse_rss_or_atom(text)
                for entry in entries:
                    normalized = normalize_article(entry)
                    if not normalized["content"]:
                        skipped += 1
                        continue
                    article_id = _save_article_if_new(session, src, normalized)
                    if article_id is not None:
                        new_ids.append(article_id)
                        success += 1
                    else:
                        skipped += 1
            elif src.type is SourceType.HTML:
                html = client.get_text(src.url)
                title, content = parse_html_with_readability(html)
                normalized = normalize_article({
                    "title": title,
                    "content_html": content,
                    "url": src.url,
                    "authors": None,
                    "published_at": None,
                })
                if not normalized["content"]:
                    skipped += 1
                else:
                    article_id = _save_article_if_new(session, src, normalized)
                    if article_id is not None:
                        new_ids.append(article_id)
                        success += 1
                    else:
                        skipped += 1
            else:
                logger.warning("Unknown source type; skipping", extra={"source_id": src.id, "type": src.type.value})
                skipped += 1
        except Exception as exc:  # noqa: BLE001
            errors += 1
            logger.exception("Failed processing source", extra={"source_id": src.id, "url": src.url, "error": str(exc)})
        finally:
            session.commit()

    logger.info(
        "Ingest completed",
        extra={
            "sources": len(sources),
            "new_articles": success,
            "skipped": skipped,
            "errors": errors,
        },
    )
    return new_ids


# --------------------------------- Celery task ---------------------------------


def _task_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap with Celery task if celery is available; otherwise pass-through."""
    try:
        from celery import shared_task  # type: ignore

        return shared_task(name="app.tasks.pull_sources")(func)  # type: ignore[return-value]
    except Exception:
        return func


@_task_decorator
def pull_sources() -> list[int]:
    """Celery task entrypoint that creates a session and ingests sources."""
    session = create_session_from_env()
    try:
        return ingest_sources(session=session)
    finally:
        session.close()


__all__ = [
    "Source",
    "SourceType",
    "Article",
    "ingest_sources",
    "pull_sources",
    "get_default_http_client",
    "HttpClient",
]

