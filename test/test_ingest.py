import os
from typing import Optional

import pytest


from app.tasks import ingest
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, select


class FakeHttpClient(ingest.HttpClient):
    def __init__(self, mapping: dict[str, str]) -> None:
        super().__init__(timeout_seconds=0.1, retries=1, backoff_base=0.0)
        self.mapping = mapping

    def get_text(self, url: str, *, headers: Optional[dict[str, str]] = None) -> str:  # type: ignore[override]
        if url not in self.mapping:
            raise ingest.HttpError(f"unknown url {url}")
        return self.mapping[url]


@pytest.fixture()
def session(tmp_path):
    db_path = tmp_path / "test.sqlite"
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    ingest.Base.metadata.create_all(engine)  # type: ignore[attr-defined]
    sess = Session(engine, future=True)
    try:
        yield sess
    finally:
        sess.close()


def test_ingest_rss_and_html(session: Session):
    # Prepare sources
    rss_src = ingest.Source(type=ingest.SourceType.RSS, url="https://example.com/rss", active=True)
    html_src = ingest.Source(type=ingest.SourceType.HTML, url="https://example.com/page", active=True)
    session.add_all([rss_src, html_src])
    session.commit()

    # Fake contents
    rss_xml = (
        """
        <rss version="2.0">
          <channel>
            <title>Example Feed</title>
            <link>https://example.com/</link>
            <description>Test</description>
            <item>
              <title>First Post</title>
              <link>https://example.com/posts/1</link>
              <description><![CDATA[<p>Hello <b>world</b> one.</p>]]></description>
              <pubDate>Wed, 01 Jan 2020 00:00:00 GMT</pubDate>
            </item>
            <item>
              <title>Second Post</title>
              <link>https://example.com/posts/2</link>
              <description><![CDATA[<p>Hello <i>world</i> two.</p>]]></description>
              <pubDate>Thu, 02 Jan 2020 00:00:00 GMT</pubDate>
            </item>
          </channel>
        </rss>
        """
    ).strip()

    html_page = (
        """
        <html><head><title>Example Page</title></head>
        <body>
          <article>
            <h1>Header</h1>
            <p>This is a test article body with enough text to make a lead.</p>
          </article>
        </body></html>
        """
    ).strip()

    client = FakeHttpClient({
        "https://example.com/rss": rss_xml,
        "https://example.com/page": html_page,
    })

    # First run: should create 3 articles
    new_ids = ingest.ingest_sources(session, http_client=client)
    assert isinstance(new_ids, list)
    assert len(new_ids) == 3

    # Second run: idempotent, no new articles
    again_ids = ingest.ingest_sources(session, http_client=client)
    assert again_ids == []

    # Validate stored articles
    articles = session.execute(select(ingest.Article)).scalars().all()
    assert len(articles) == 3
    # Ensure unique content hashes and proper normalization
    hashes = {a.content_hash for a in articles}
    assert len(hashes) == 3
    for a in articles:
        assert a.title
        assert a.content
        assert a.lead
        assert len(a.lead) <= 240
        assert a.url

    # Counts per source
    rss_count = session.execute(select(ingest.Article).where(ingest.Article.source_id == rss_src.id)).scalars().all()
    html_count = session.execute(select(ingest.Article).where(ingest.Article.source_id == html_src.id)).scalars().all()
    assert len(rss_count) == 2
    assert len(html_count) == 1

