"""
FastAPI service providing endpoints for managing sources, clients, channels (schedules),
and triggering the content pipeline (ingest/translate/rewrite/moderate/publish).

Features:
- JWT auth (HS256) with admin role for mutating endpoints
- Pydantic v2 schemas with validation and examples
- Pagination and basic filters
- Swagger/OpenAPI examples
"""

from __future__ import annotations

import base64
import hmac
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Generator, Optional

from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, HttpUrl, ValidationError
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from app.tasks.ingest import Base, Source, SourceType, create_session_from_env
from app.tasks.publish import Platform, Schedule, Post, PostStatus
from app.tasks import ingest as ingest_task
from app.tasks import translate as translate_task
from app.tasks import rewrite as rewrite_task
from app.tasks import moderate as moderate_task
from app.tasks import publish as publish_task
from app.utils.logging import configure_json_logging


logger = logging.getLogger("services.api")
app = FastAPI(title="Content Pipeline API", version="0.1.0")


# ----------------------------- Auth (JWT HS256) -----------------------------


class User(BaseModel):
    sub: str
    role: str = "user"


bearer = HTTPBearer(auto_error=False)


def _b64url_decode(data: str) -> bytes:
    pad = '=' * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + pad)


def _decode_jwt_hs256(token: str, secret: str) -> dict[str, Any]:
    try:
        header_b64, payload_b64, sig_b64 = token.split('.')
    except ValueError as exc:  # noqa: BLE001
        raise HTTPException(status_code=401, detail="Invalid token format") from exc
    signing_input = f"{header_b64}.{payload_b64}".encode()
    expected = hmac.new(secret.encode(), signing_input, 'sha256').digest()
    try:
        signature = _b64url_decode(sig_b64)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=401, detail="Invalid token signature") from exc
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=401, detail="Invalid token signature")
    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=401, detail="Invalid token payload") from exc
    # Optional exp check
    exp = payload.get("exp")
    if isinstance(exp, (int, float)):
        if datetime.now(timezone.utc).timestamp() > float(exp):
            raise HTTPException(status_code=401, detail="Token expired")
    return payload


def get_current_user(
    cred: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
) -> User:
    if cred is None or not cred.scheme.lower() == "bearer":
        raise HTTPException(status_code=401, detail="Missing bearer token")
    secret = os.getenv("JWT_SECRET", "devsecret")
    payload = _decode_jwt_hs256(cred.credentials, secret)
    try:
        return User.model_validate({"sub": payload.get("sub", "user"), "role": payload.get("role", "user")})
    except ValidationError:
        raise HTTPException(status_code=401, detail="Invalid token claims")


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return user


# ------------------------------- DB dependency -------------------------------


def get_session() -> Generator[Session, None, None]:
    session = create_session_from_env()
    try:
        yield session
    finally:
        session.close()


@app.on_event("startup")
def _on_startup() -> None:
    """Configure JSON logging on startup."""
    configure_json_logging()



# --------------------------------- Schemas -----------------------------------


class PageMeta(BaseModel):
    total: int
    offset: int
    limit: int


class PageSources(BaseModel):
    items: list[dict]
    meta: PageMeta


class SourceIn(BaseModel):
    type: SourceType = Field(examples=["rss", "atom", "html"])
    url: HttpUrl = Field(examples=["https://example.com/rss.xml"])  # type: ignore[assignment]
    active: bool = True


class SourceOut(BaseModel):
    id: int
    type: SourceType
    url: str
    active: bool

    @staticmethod
    def from_orm_obj(obj: Source) -> "SourceOut":
        return SourceOut(id=obj.id, type=obj.type, url=obj.url, active=obj.active)


class PageClients(BaseModel):
    items: list[dict]
    meta: PageMeta


class ClientIn(BaseModel):
    brief: str = Field(min_length=1, max_length=4000, examples=["B2B SaaS, продвигать демо-версию"])
    tone: Optional[str] = Field(default=None, max_length=64, examples=["профессиональный"])
    required_hashtags: Optional[list[str]] = Field(default=None, examples=[["#SaaS", "#B2B"]])
    forbidden_words: Optional[list[str]] = Field(default=None, examples=[["бесплатно", "скидка"]])


class ClientOut(BaseModel):
    id: int
    brief: str
    tone: Optional[str]
    required_hashtags: Optional[list[str]]
    forbidden_words: Optional[list[str]]


class PageChannels(BaseModel):
    items: list[dict]
    meta: PageMeta


class ChannelIn(BaseModel):
    platform: Platform = Field(examples=["telegram", "vk"])
    cron: str = Field(examples=["*/10 * * * *"])
    timezone: str = Field(default="UTC", examples=["Europe/Moscow"])
    daily_limit: int = Field(default=24, ge=1, le=200)


class ChannelOut(BaseModel):
    id: int
    platform: Platform
    cron: str
    timezone: str
    daily_limit: int


class PipelineIngestOut(BaseModel):
    article_ids: list[int]


class PipelineTranslateIn(BaseModel):
    article_id: int = Field(examples=[1])
    lang_target: str = Field(examples=["ru", "en"])


class PipelineTranslateOut(BaseModel):
    translation_id: int


class PipelineRewriteIn(BaseModel):
    translation_id: int
    client_id: int


class PipelineRewriteOut(BaseModel):
    rewrite_id: int


class PipelineModerateIn(BaseModel):
    rewrite_id: int


class PipelineModerateOut(BaseModel):
    moderation_id: int


class PipelineScheduleOut(BaseModel):
    post_ids: list[int]


class PipelinePublishIn(BaseModel):
    post_id: int


class PipelinePublishOut(BaseModel):
    post_id: int
    status: PostStatus
    platform_post_id: Optional[str]


# --------------------------------- Endpoints ---------------------------------


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


# ------------------------------- Sources CRUD ---------------------------------


@app.get("/sources", response_model=PageSources)
def list_sources(
    active: Optional[bool] = Query(default=None),
    q: Optional[str] = Query(default=None, description="Filter by URL substring"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
) -> PageSources:
    stmt = select(Source)
    if active is not None:
        stmt = stmt.where(Source.active.is_(active))
    if q:
        stmt = stmt.where(Source.url.contains(q))
    total = session.execute(select(func.count(Source.id)).select_from(stmt.subquery())).scalar_one()
    rows = session.execute(stmt.order_by(Source.id.desc()).offset(offset).limit(limit)).scalars().all()
    items = [SourceOut.from_orm_obj(r).model_dump() for r in rows]
    return PageSources(items=items, meta=PageMeta(total=total, offset=offset, limit=limit))


@app.post("/sources", response_model=SourceOut, status_code=201)
def create_source(payload: SourceIn, session: Session = Depends(get_session), _: User = Depends(require_admin)) -> SourceOut:
    obj = Source(type=payload.type, url=str(payload.url), active=payload.active)
    session.add(obj)
    session.commit()
    session.refresh(obj)
    return SourceOut.from_orm_obj(obj)


@app.patch("/sources/{source_id}", response_model=SourceOut)
def update_source(
    source_id: int,
    payload: SourceIn,
    session: Session = Depends(get_session),
    _: User = Depends(require_admin),
) -> SourceOut:
    obj = session.get(Source, source_id)
    if obj is None:
        raise HTTPException(status_code=404, detail="Source not found")
    obj.type = payload.type
    obj.url = str(payload.url)
    obj.active = payload.active
    session.add(obj)
    session.commit()
    return SourceOut.from_orm_obj(obj)


@app.delete("/sources/{source_id}", status_code=204)
def delete_source(source_id: int, session: Session = Depends(get_session), _: User = Depends(require_admin)) -> Response:
    obj = session.get(Source, source_id)
    if obj is None:
        raise HTTPException(status_code=404, detail="Source not found")
    session.delete(obj)
    session.commit()
    return Response(status_code=204)


# ------------------------------- Clients CRUD ---------------------------------


@app.get("/clients", response_model=PageClients)
def list_clients(
    q: Optional[str] = Query(default=None, description="Search in brief"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
) -> PageClients:
    from app.tasks.rewrite import Client  # imported here to avoid circular import at module load

    stmt = select(Client)
    if q:
        stmt = stmt.where(Client.brief.contains(q))
    total = session.execute(select(func.count(Client.id)).select_from(stmt.subquery())).scalar_one()
    rows = session.execute(stmt.order_by(Client.id.desc()).offset(offset).limit(limit)).scalars().all()
    items = [
        ClientOut(id=r.id, brief=r.brief, tone=r.tone, required_hashtags=r.required_hashtags, forbidden_words=r.forbidden_words).model_dump()
        for r in rows
    ]
    return PageClients(items=items, meta=PageMeta(total=total, offset=offset, limit=limit))


@app.post("/clients", response_model=ClientOut, status_code=201)
def create_client(
    payload: ClientIn,
    session: Session = Depends(get_session),
    _: User = Depends(require_admin),
) -> ClientOut:
    from app.tasks.rewrite import Client

    obj = Client(
        brief=payload.brief,
        tone=payload.tone,
        required_hashtags=payload.required_hashtags,
        forbidden_words=payload.forbidden_words,
    )
    session.add(obj)
    session.commit()
    session.refresh(obj)
    return ClientOut(id=obj.id, brief=obj.brief, tone=obj.tone, required_hashtags=obj.required_hashtags, forbidden_words=obj.forbidden_words)


@app.patch("/clients/{client_id}", response_model=ClientOut)
def update_client(
    client_id: int,
    payload: ClientIn,
    session: Session = Depends(get_session),
    _: User = Depends(require_admin),
) -> ClientOut:
    from app.tasks.rewrite import Client

    obj = session.get(Client, client_id)
    if obj is None:
        raise HTTPException(status_code=404, detail="Client not found")
    obj.brief = payload.brief
    obj.tone = payload.tone
    obj.required_hashtags = payload.required_hashtags
    obj.forbidden_words = payload.forbidden_words
    session.add(obj)
    session.commit()
    return ClientOut(id=obj.id, brief=obj.brief, tone=obj.tone, required_hashtags=obj.required_hashtags, forbidden_words=obj.forbidden_words)


@app.delete("/clients/{client_id}", status_code=204)
def delete_client(client_id: int, session: Session = Depends(get_session), _: User = Depends(require_admin)) -> Response:
    from app.tasks.rewrite import Client

    obj = session.get(Client, client_id)
    if obj is None:
        raise HTTPException(status_code=404, detail="Client not found")
    session.delete(obj)
    session.commit()
    return Response(status_code=204)


# ------------------------------ Channels (Schedules) --------------------------


@app.get("/channels", response_model=PageChannels)
def list_channels(
    platform: Optional[Platform] = Query(default=None),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
) -> PageChannels:
    stmt = select(Schedule)
    if platform is not None:
        stmt = stmt.where(Schedule.platform == platform)
    total = session.execute(select(func.count(Schedule.id)).select_from(stmt.subquery())).scalar_one()
    rows = session.execute(stmt.order_by(Schedule.id.desc()).offset(offset).limit(limit)).scalars().all()
    items = [ChannelOut(id=r.id, platform=r.platform, cron=r.cron, timezone=r.timezone, daily_limit=r.daily_limit).model_dump() for r in rows]
    return PageChannels(items=items, meta=PageMeta(total=total, offset=offset, limit=limit))


@app.post("/channels", response_model=ChannelOut, status_code=201)
def create_channel(payload: ChannelIn, session: Session = Depends(get_session), _: User = Depends(require_admin)) -> ChannelOut:
    obj = Schedule(platform=payload.platform, cron=payload.cron, timezone=payload.timezone, daily_limit=payload.daily_limit)
    session.add(obj)
    session.commit()
    session.refresh(obj)
    return ChannelOut(id=obj.id, platform=obj.platform, cron=obj.cron, timezone=obj.timezone, daily_limit=obj.daily_limit)


@app.patch("/channels/{channel_id}", response_model=ChannelOut)
def update_channel(
    channel_id: int,
    payload: ChannelIn,
    session: Session = Depends(get_session),
    _: User = Depends(require_admin),
) -> ChannelOut:
    obj = session.get(Schedule, channel_id)
    if obj is None:
        raise HTTPException(status_code=404, detail="Channel not found")
    obj.platform = payload.platform
    obj.cron = payload.cron
    obj.timezone = payload.timezone
    obj.daily_limit = payload.daily_limit
    session.add(obj)
    session.commit()
    return ChannelOut(id=obj.id, platform=obj.platform, cron=obj.cron, timezone=obj.timezone, daily_limit=obj.daily_limit)


@app.delete("/channels/{channel_id}", status_code=204)
def delete_channel(channel_id: int, session: Session = Depends(get_session), _: User = Depends(require_admin)) -> Response:
    obj = session.get(Schedule, channel_id)
    if obj is None:
        raise HTTPException(status_code=404, detail="Channel not found")
    session.delete(obj)
    session.commit()
    return Response(status_code=204)


# --------------------------------- Pipeline ----------------------------------


@app.post("/pipeline/ingest", response_model=PipelineIngestOut)
def pipeline_ingest(session: Session = Depends(get_session), _: User = Depends(require_admin)) -> PipelineIngestOut:
    ids = ingest_task.ingest_sources(session)
    return PipelineIngestOut(article_ids=ids)


@app.post("/pipeline/translate", response_model=PipelineTranslateOut)
def pipeline_translate(payload: PipelineTranslateIn, session: Session = Depends(get_session), _: User = Depends(require_admin)) -> PipelineTranslateOut:
    tr_id = translate_task.translate_article_core(session, article_id=payload.article_id, lang_target=payload.lang_target)
    return PipelineTranslateOut(translation_id=tr_id)


@app.post("/pipeline/rewrite", response_model=PipelineRewriteOut)
def pipeline_rewrite(payload: PipelineRewriteIn, session: Session = Depends(get_session), _: User = Depends(require_admin)) -> PipelineRewriteOut:
    rw_id = rewrite_task.rewrite_for_client_core(session, translation_id=payload.translation_id, client_id=payload.client_id)
    return PipelineRewriteOut(rewrite_id=rw_id)


@app.post("/pipeline/moderate", response_model=PipelineModerateOut)
def pipeline_moderate(payload: PipelineModerateIn, session: Session = Depends(get_session), _: User = Depends(require_admin)) -> PipelineModerateOut:
    mod_id = moderate_task.moderate_rewrite_core(session, rewrite_id=payload.rewrite_id)
    return PipelineModerateOut(moderation_id=mod_id)


@app.post("/pipeline/schedule", response_model=PipelineScheduleOut)
def pipeline_schedule(session: Session = Depends(get_session), _: User = Depends(require_admin)) -> PipelineScheduleOut:
    post_ids = publish_task.schedule_posts_core(session)
    return PipelineScheduleOut(post_ids=post_ids)


@app.post("/pipeline/publish", response_model=PipelinePublishOut)
def pipeline_publish(payload: PipelinePublishIn, session: Session = Depends(get_session), _: User = Depends(require_admin)) -> PipelinePublishOut:
    publish_task.publish_post_core(session, post_id=payload.post_id)
    post = session.get(Post, payload.post_id)
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found after publish")
    return PipelinePublishOut(post_id=post.id, status=post.status, platform_post_id=post.platform_post_id)


# ------------------------------- Swagger examples ----------------------------


@app.get("/", include_in_schema=False)
def index() -> Response:
    return Response(content="OK", media_type="text/plain")

