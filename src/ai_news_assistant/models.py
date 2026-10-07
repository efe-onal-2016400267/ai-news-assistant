from typing import Annotated

from pydantic import AwareDatetime, BaseModel, Field, HttpUrl, StringConstraints


class Article(BaseModel):
    """A news article as ingested from the source."""

    id: str  # Stable hash of the canonical URL
    url: HttpUrl
    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    source: str  # e.g. BBC, Guardian
    published_at: AwareDatetime | None = None
    summary: str | None = None  # Feed teaser
    text: str | None = None  # Extracted full text, if available
    tags: list[str] = Field(default_factory=list)  # Tags for our later metadata search
