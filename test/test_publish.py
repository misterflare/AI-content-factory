import os
from typing import Any

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.tasks import ingest, translate, rewrite, moderate, publish


class FakeRequester(translate.Requester):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.responses: dict[str, list[dict[str, Any]]] = {}

    def set_response(self, url_prefix: str, responses: list[dict[str, Any]]) -> None:
        self.responses[url_prefix] = responses

    def post_json(self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float) -> dict[str, Any]:  # type: ignore[override]
        self.calls.append((url, json))
        for prefix, resps in self.responses.items():
            if url.startswith(prefix):
                if not resps:
                    raise AssertionError(f"No more responses for prefix {prefix}")
                return resps.pop(0)
        raise AssertionError(f"Unexpected URL: {url}")


@pytest.fixture()
def session(tmp_path):
    db_path = tmp_path / "test_publish.sqlite"
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    ingest.Base.metadata.create_all(engine)
    translate.Base.metadata.create_all(engine)
    rewrite.Base.metadata.create_all(engine)
    moderate.Base.metadata.create_all(engine)
    publish.Base.metadata.create_all(engine)
    sess = Session(engine, future=True)
    try:
        yield sess
    finally:
        sess.close()


def _prepare_approved_rewrite(session: Session, text: str = "Hello TG/VK!") -> rewrite.Rewrite:
    src = ingest.Source(type=ingest.SourceType.HTML, url="https://example.com", active=True)
    art = ingest.Article(
        source=src,
        title="T",
        content="C",
        lead="L",
        authors=None,
        published_at=None,
        url="https://example.com/x",
        content_hash="h-pub",
    )
    tr = translate.Translation(article=art, lang="ru", provider="deepl", text=text)
    client = rewrite.Client(brief="", tone=None, required_hashtags=None, forbidden_words=None)
    rw = rewrite.Rewrite(translation=tr, client=client, text=text)
    mod = moderate.Moderation(target_type=moderate.ModerationTarget.REWRITE, target_id=rw.id, decision=moderate.ModerationDecision.OK, reasons=[])
    session.add_all([src, art, tr, client, rw, mod])
    session.commit()
    return rw


def test_schedule_and_publish_telegram(session: Session, monkeypatch: pytest.MonkeyPatch):
    rw = _prepare_approved_rewrite(session, text="Привет, Telegram!")

    sched = publish.Schedule(platform=publish.Platform.TELEGRAM, cron="*/5 * * * *", timezone="UTC", daily_limit=2)
    session.add(sched)
    session.commit()

    new_ids = publish.schedule_posts_core(session)
    assert len(new_ids) == 1
    post = session.get(publish.Post, new_ids[0])
    assert post is not None
    assert post.platform == publish.Platform.TELEGRAM
    assert post.status == publish.PostStatus.SCHEDULED
    assert post.text.startswith("Привет")

    # Mock Telegram sendMessage
    fake = FakeRequester()
    bot_token = "test-token"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", bot_token)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-1001")
    fake.set_response(f"https://api.telegram.org/bot{bot_token}", [{"ok": True, "result": {"message_id": 42}}])

    publish.publish_post_core(session, post_id=post.id, requester=fake)
    post2 = session.get(publish.Post, post.id)
    assert post2 is not None
    assert post2.status == publish.PostStatus.PUBLISHED
    assert post2.platform_post_id == "42"

    # Check payload formatting
    called = fake.calls[-1]
    assert called[0].endswith("/sendMessage") or called[0].endswith("/sendPhoto")
    assert called[1]["chat_id"] == "-1001"
    assert "text" in called[1] or "caption" in called[1]


def test_schedule_and_publish_vk(session: Session, monkeypatch: pytest.MonkeyPatch):
    rw = _prepare_approved_rewrite(session, text="Привет, ВК!")

    sched = publish.Schedule(platform=publish.Platform.VK, cron="*/10 * * * *", timezone="UTC", daily_limit=2)
    session.add(sched)
    session.commit()

    new_ids = publish.schedule_posts_core(session)
    assert len(new_ids) == 1
    post = session.get(publish.Post, new_ids[0])
    assert post is not None
    assert post.platform == publish.Platform.VK

    # Mock VK wall.post
    fake = FakeRequester()
    monkeypatch.setenv("VK_ACCESS_TOKEN", "vk-token")
    monkeypatch.setenv("VK_GROUP_ID", "123")
    fake.set_response("https://api.vk.com/method/wall.post", [{"response": {"post_id": 777}}])

    publish.publish_post_core(session, post_id=post.id, requester=fake)
    post2 = session.get(publish.Post, post.id)
    assert post2 is not None
    assert post2.status == publish.PostStatus.PUBLISHED
    assert post2.platform_post_id == "777"

    # Check payload formatting
    url, body = fake.calls[-1]
    assert url.startswith("https://api.vk.com/method/wall.post")
    assert body["owner_id"] == -123
    assert body["from_group"] == 1
    assert "message" in body

