from ai_news_assistant.ingestion.base import Fetcher
from ai_news_assistant.models import Article


class FakeFetcher:
    """Satisfies Fetcher structurally, without inheriting from it."""

    def fetch(self) -> list[Article]:
        return [
            Article(id="abc", url="https://example.com/a", title="A headline", source="Example")
        ]


def collect(fetchers: list[Fetcher]) -> list[Article]:
    return [article for fetcher in fetchers for article in fetcher.fetch()]


# isinstance on a runtime_checkable protocol only checks that `fetch` exists,
# not its signature or return type; mypy covers those in src.


def test_fake_fetcher_satisfies_protocol() -> None:
    assert isinstance(FakeFetcher(), Fetcher)


def test_fetch_returns_list_of_articles() -> None:
    articles = FakeFetcher().fetch()

    assert isinstance(articles, list)
    assert all(isinstance(article, Article) for article in articles)


def test_object_without_fetch_is_not_a_fetcher() -> None:
    assert not isinstance(object(), Fetcher)


def test_callers_use_any_fetcher() -> None:
    articles = collect([FakeFetcher(), FakeFetcher()])

    assert [a.title for a in articles] == ["A headline", "A headline"]
