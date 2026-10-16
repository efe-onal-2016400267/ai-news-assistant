from pydantic import BaseModel, Field

from ai_news_assistant.models import NonEmptyStr


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
