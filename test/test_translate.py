import os
from typing import Any

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.tasks import ingest, translate


class FakeRequester(translate.Requester):
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
    db_path = tmp_path / "test_translate.sqlite"
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    ingest.Base.metadata.create_all(engine)  # Sources/Articles
    translate.Base.metadata.create_all(engine)  # Translations
    sess = Session(engine, future=True)
    try:
        yield sess
    finally:
        sess.close()


def test_translate_success_and_idempotent(session: Session, monkeypatch: pytest.MonkeyPatch):
    # Create article
    src = ingest.Source(type=ingest.SourceType.HTML, url="https://example.com", active=True)
    art = ingest.Article(
        source=src,
        title="Hello",
        content="Hello world content for translation.",
        lead="Hello world",
        authors=None,
        published_at=None,
        url="https://example.com/x",
        content_hash="hash1",
    )
    session.add_all([src, art])
    session.commit()

    # Configure provider and fake requester (DeepL shape)
    monkeypatch.setenv("LLM_PROVIDER", "deepl")
    # Speed up retries if any occur in code paths
    monkeypatch.setenv("TRANSLATE_BACKOFF", "0")
    monkeypatch.setenv("TRANSLATE_RETRIES", "2")
    fake = FakeRequester(responses=[{"translations": [{"text": "Привет мир контент перевода."}]}])

    tr_id_1 = translate.translate_article_core(session, article_id=art.id, lang_target="ru", requester=fake)
    assert isinstance(tr_id_1, int)

    # Idempotent: second call returns same row id
    tr_id_2 = translate.translate_article_core(session, article_id=art.id, lang_target="ru", requester=fake)
    assert tr_id_2 == tr_id_1

    # Validate DB
    tr_rows = session.execute(select(translate.Translation)).scalars().all()
    assert len(tr_rows) == 1
    assert tr_rows[0].text.startswith("Привет")
    assert tr_rows[0].provider == "deepl"


def test_translate_retries_then_success(session: Session, monkeypatch: pytest.MonkeyPatch):
    # Article
    src = ingest.Source(type=ingest.SourceType.HTML, url="https://example.com", active=True)
    art = ingest.Article(
        source=src,
        title="Hello",
        content="Text",
        lead="Text",
        authors=None,
        published_at=None,
        url="https://example.com/x",
        content_hash="hash2",
    )
    session.add_all([src, art])
    session.commit()

    monkeypatch.setenv("LLM_PROVIDER", "google")
    monkeypatch.setenv("TRANSLATE_BACKOFF", "0")
    monkeypatch.setenv("TRANSLATE_RETRIES", "2")
    # Fail once, then succeed with Google response shape
    fake = FakeRequester(
        responses=[{"data": {"translations": [{"translatedText": "Hola"}]}}],
        fail_times=1,
    )
    tr_id = translate.translate_article_core(session, article_id=art.id, lang_target="es", requester=fake)
    assert isinstance(tr_id, int)
    row = session.execute(select(translate.Translation).where(translate.Translation.id == tr_id)).scalar_one()
    assert row.text == "Hola"
    assert row.provider == "google"


def test_translate_fails_after_retries(session: Session, monkeypatch: pytest.MonkeyPatch):
    # Article
    src = ingest.Source(type=ingest.SourceType.HTML, url="https://example.com", active=True)
    art = ingest.Article(
        source=src,
        title="Hello",
        content="Short",
        lead="Short",
        authors=None,
        published_at=None,
        url="https://example.com/x",
        content_hash="hash3",
    )
    session.add_all([src, art])
    session.commit()

    monkeypatch.setenv("LLM_PROVIDER", "yandex")
    monkeypatch.setenv("TRANSLATE_BACKOFF", "0")
    monkeypatch.setenv("TRANSLATE_RETRIES", "2")
    # Always fail
    fake = FakeRequester(responses=[], fail_times=2)

    with pytest.raises(translate.TranslationError):
        translate.translate_article_core(session, article_id=art.id, lang_target="de", requester=fake)

    # Ensure nothing saved
    rows = session.execute(select(translate.Translation).where(translate.Translation.article_id == art.id)).scalars().all()
    assert len(rows) == 0
