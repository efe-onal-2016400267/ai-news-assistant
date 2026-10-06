
from pydantic import AwareDatetime, BaseModel, Field, HttpUrl


class Article(BaseModel):
    """ Base article class, so we can read articles without type errors. """
    id: str # Stable hash of the canonical URL
    url: HttpUrl
    title: str = Field(min_length=5)
    source: str # e.g. BBC, Guardian
    published_at: AwareDatetime | None
    summary: str | None = None # Feed teaser
    text: str | None = None # Extracted full text, if available
    tags: list[str] = Field(default_factory=list) # Tags for our later metadata search
