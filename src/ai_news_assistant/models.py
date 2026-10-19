from typing import Annotated, Self

from pydantic import AwareDatetime, BaseModel, Field, HttpUrl, StringConstraints, model_validator


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
    # None: extraction not tried yet, False: it failed, True: it succeeded.
    extraction_successful: bool | None = None
    extracted_at: AwareDatetime | None = None  # Set only when extraction succeeded

    @model_validator(mode="after")
    def _check_extraction_state(self) -> Self:
        if self.extracted_at is not None and self.extraction_successful is not True:
            raise ValueError("extracted_at is only set when extraction_successful is True")
        if self.extraction_successful is True and (self.text is None or not self.text.strip()):
            raise ValueError("non-empty text is required when extraction_successful is True")
        return self


# Citation models
# Source and Answer will deterministically validate model output
# before creating the actual citation.
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Source(BaseModel):
    """One entry in the Sources block, filled from the store (never by the model)."""

    marker: int = Field(ge=1)  # the [n] in the answer text this source backs
    article_id: NonEmptyStr  # internal only; render() will never print it
    title: NonEmptyStr
    source: str  # e.g. Ars Technica
    published_at: AwareDatetime | None = None


class Answer(BaseModel):
    """The checked answer that the CLI, API, and evals receive."""

    text: str
    sources: list[Source] = Field(default_factory=list)
    grounded: bool


class ModelAnswer(BaseModel):
    """The answer as the model writes it. Nothing here is trusted yet."""

    text: str = Field(
        description=(
            "The answer for the user. After each claim taken from the article, put a marker "
            "like [1] or [2]. Never write the article ids in the text."
        )
    )
    citations: list[NonEmptyStr] = Field(
        default_factory=list,
        description=(
            "Article ids from tool results. The first id backs marker [1], the second "
            "backs [2], and so on. Only ids that appeared in tool results."
        ),
    )
    grounded: bool = Field(
        description="True if the answer is based on retrieved articles, false otherwise."
    )
