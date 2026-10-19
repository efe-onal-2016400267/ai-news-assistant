from ai_news_assistant.models import Answer


def render(answer: Answer) -> str:
    """Add the sources at the end of the answer."""

    if not answer.sources:
        return answer.text

    text = answer.text + "\n"
    for source in answer.sources:
        text += "[" + str(source.marker) + "]: " + source.title + ", " + source.source
        if not source.published_at:
            date_text = "Date not found"
        else:
            date_text = f"{source.published_at.day} {source.published_at:%b %Y}"
        text += ", " + date_text + "\n"
    return text
