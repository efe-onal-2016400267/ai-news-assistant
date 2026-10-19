from ai_news_assistant.agent.base import format_date


def test_format_date_none_is_unknown() -> None:
    assert format_date(None) == "date unknown"


def test_format_date_day_month_year() -> None:
    assert format_date("2026-10-10T09:30:00+00:00") == "10 Oct 2026"


def test_format_date_single_digit_day_not_padded() -> None:
    assert format_date("2026-10-05T00:00:00+00:00") == "5 Oct 2026"
