import os
from typing import Any

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.tasks import ingest, translate, rewrite


class FakeRequester(rewrite.Requester):
    def __init__(self, responses: list[dict[str, Any]] | None = None, *, fail_times: int = 0, error: Exception | None = None) -> None:
        self.responses = responses or []
        self.fail_times = fail_times
        self.error = error or RuntimeError("httpx error simulated")
        self.calls = 0

    def post_json(self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float) -> dict[str, Any]:  # type: ignore[override]
        self.calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.error
        if not self.responses:
            raise AssertionError("No more responses configured")
        return self.responses.pop(0)


@pytest.fixture()
def session(tmp_path):
    db_path = tmp_path / "test_rewrite.sqlite"
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    # Ensure all tables exist
    ingest.Base.metadata.create_all(engine)
    translate.Base.metadata.create_all(engine)
    rewrite.Base.metadata.create_all(engine)
    sess = Session(engine, future=True)
    try:
        yield sess
    finally:
        sess.close()


def _prepare_article_and_translation(session: Session) -> tuple[ingest.Article, translate.Translation, rewrite.Client]:
    src = ingest.Source(type=ingest.SourceType.HTML, url="https://example.com", active=True)
    art = ingest.Article(
        source=src,
        title="Hello",
        content="This is a long text for rewriting into a short social post.",
        lead="This is a long text",
        authors=None,
        published_at=None,
        url="https://example.com/x",
        content_hash="h0",
    )
    tr = translate.Translation(article=art, lang="ru", provider="deepl", text="Это пример перевода длинного текста для переписывания.")
    client = rewrite.Client(
        brief="B2B SaaS, продвигать бесплатную демо-версию",
        tone="профессиональный",
        required_hashtags=["#SaaS", "#B2B", "#Demo"],
        forbidden_words=["бесплатно", "скидка"],
    )
    session.add_all([src, art, tr, client])
    session.commit()
    return art, tr, client


def test_rewrite_success_and_idempotent(session: Session, monkeypatch: pytest.MonkeyPatch):
    _, tr, client = _prepare_article_and_translation(session)

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_RETRIES", "2")
    monkeypatch.setenv("LLM_BACKOFF", "0")
    monkeypatch.setenv("REWRITE_MAX_TOKENS", "128")

    fake = FakeRequester(responses=[{"choices": [{"message": {"content": "Короткий пост #SaaS #B2B. Зарегистрируйтесь!"}}]}])

    rw_id_1 = rewrite.rewrite_for_client_core(session, translation_id=tr.id, client_id=client.id, requester=fake)
    assert isinstance(rw_id_1, int)

    # Idempotent: same ids on second run
    rw_id_2 = rewrite.rewrite_for_client_core(session, translation_id=tr.id, client_id=client.id, requester=fake)
    assert rw_id_2 == rw_id_1

    row = session.execute(select(rewrite.Rewrite).where(rewrite.Rewrite.id == rw_id_1)).scalar_one()
    assert "#SaaS" in row.text


def test_rewrite_retries_then_success(session: Session, monkeypatch: pytest.MonkeyPatch):
    _, tr, client = _prepare_article_and_translation(session)

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_RETRIES", "2")
    monkeypatch.setenv("LLM_BACKOFF", "0")

    fake = FakeRequester(
        responses=[{"choices": [{"message": {"content": "Ещё один короткий пост #Demo. Попробуйте демо!"}}]}],
        fail_times=1,
    )

    rw_id = rewrite.rewrite_for_client_core(session, translation_id=tr.id, client_id=client.id, requester=fake)
    assert isinstance(rw_id, int)
    row = session.execute(select(rewrite.Rewrite).where(rewrite.Rewrite.id == rw_id)).scalar_one()
    assert "демо" in row.text.lower()


def test_rewrite_fails_after_retries(session: Session, monkeypatch: pytest.MonkeyPatch):
    _, tr, client = _prepare_article_and_translation(session)

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_RETRIES", "2")
    monkeypatch.setenv("LLM_BACKOFF", "0")

    fake = FakeRequester(responses=[], fail_times=2)

    with pytest.raises(rewrite.LLMError):
        rewrite.rewrite_for_client_core(session, translation_id=tr.id, client_id=client.id, requester=fake)

    rows = session.execute(select(rewrite.Rewrite).where(rewrite.Rewrite.translation_id == tr.id)).scalars().all()
    assert len(rows) == 0

