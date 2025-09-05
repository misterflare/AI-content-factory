import os
from typing import Any

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.tasks import ingest, translate, rewrite, moderate


class FakeLLM(moderate.LLMClient):  # type: ignore[misc]
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs

    def generate(self, system_prompt: str, user_prompt: str, *, max_tokens: int) -> str:  # type: ignore[override]
        if not self.outputs:
            raise RuntimeError("No more outputs configured")
        return self.outputs.pop(0)


@pytest.fixture()
def session(tmp_path):
    db_path = tmp_path / "test_moderate.sqlite"
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    ingest.Base.metadata.create_all(engine)
    translate.Base.metadata.create_all(engine)
    rewrite.Base.metadata.create_all(engine)
    moderate.Base.metadata.create_all(engine)
    sess = Session(engine, future=True)
    try:
        yield sess
    finally:
        sess.close()


def _prepare_rewrite(session: Session, text: str) -> rewrite.Rewrite:
    src = ingest.Source(type=ingest.SourceType.HTML, url="https://example.com/page", active=True)
    art = ingest.Article(
        source=src,
        title="T",
        content="C",
        lead="L",
        authors=None,
        published_at=None,
        url="https://example.com/page",
        content_hash="h-moderate",
    )
    tr = translate.Translation(article=art, lang="ru", provider="deepl", text="Текст перевода")
    client = rewrite.Client(brief="", tone=None, required_hashtags=None, forbidden_words=None)
    rw = rewrite.Rewrite(translation=tr, client=client, text=text)
    session.add_all([src, art, tr, client, rw])
    session.commit()
    return rw


def test_moderation_ok(session: Session, monkeypatch: pytest.MonkeyPatch):
    # Acceptable text with source URL
    text = "Это валидный пост с источником: https://example.com/page и без стоп-слов."
    rw = _prepare_rewrite(session, text)

    monkeypatch.setenv("MODERATION_MIN_CHARS", "10")
    monkeypatch.setenv("MODERATION_MAX_CHARS", "500")
    monkeypatch.setenv("MODERATION_REQUIRE_SOURCE", "1")
    monkeypatch.setenv("MODERATION_STOP_WORDS", "")
    monkeypatch.setenv("MODERATION_ALLOWED_TOPICS", "")
    monkeypatch.setenv("MODERATION_BLOCKED_TOPICS", "")

    llm = FakeLLM(["ok"])
    mod_id = moderate.moderate_rewrite_core(session, rewrite_id=rw.id, llm=llm)
    row = session.execute(select(moderate.Moderation).where(moderate.Moderation.id == mod_id)).scalar_one()
    assert row.decision == moderate.ModerationDecision.OK
    assert row.reasons == []


def test_moderation_needs_review_due_to_stop_word(session: Session, monkeypatch: pytest.MonkeyPatch):
    text = "Это пост содержит spam и ссылку https://example.com/page"
    rw = _prepare_rewrite(session, text)

    monkeypatch.setenv("MODERATION_MIN_CHARS", "10")
    monkeypatch.setenv("MODERATION_MAX_CHARS", "500")
    monkeypatch.setenv("MODERATION_REQUIRE_SOURCE", "1")
    monkeypatch.setenv("MODERATION_STOP_WORDS", "spam")
    llm = FakeLLM(["ok"])  # Heuristics should still flag

    mod_id = moderate.moderate_rewrite_core(session, rewrite_id=rw.id, llm=llm)
    row = session.execute(select(moderate.Moderation).where(moderate.Moderation.id == mod_id)).scalar_one()
    assert row.decision == moderate.ModerationDecision.NEEDS_REVIEW
    assert any(r.startswith("stop_word:") for r in row.reasons)


def test_moderation_reject_from_llm(session: Session, monkeypatch: pytest.MonkeyPatch):
    text = "Формально всё норм, но LLM попросит reject. Ссылка: https://example.com/page"
    rw = _prepare_rewrite(session, text)

    monkeypatch.setenv("MODERATION_MIN_CHARS", "10")
    monkeypatch.setenv("MODERATION_MAX_CHARS", "500")
    monkeypatch.setenv("MODERATION_REQUIRE_SOURCE", "1")
    monkeypatch.setenv("MODERATION_STOP_WORDS", "")

    llm = FakeLLM(["reject"])  # Force reject

    mod_id = moderate.moderate_rewrite_core(session, rewrite_id=rw.id, llm=llm)
    row = session.execute(select(moderate.Moderation).where(moderate.Moderation.id == mod_id)).scalar_one()
    assert row.decision == moderate.ModerationDecision.REJECT

