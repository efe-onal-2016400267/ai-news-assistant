import ai_news_assistant


def test_package_imports() -> None:
    assert hasattr(ai_news_assistant, "main")
