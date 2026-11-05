from pathlib import Path

import pytest

from ai_news_assistant.api.settings import REQUIRED_ENV, db_uri, preflight


@pytest.fixture
def db_file(tmp_path: Path) -> Path:
    db = tmp_path / "news.db"
    db.touch()
    return db


@pytest.fixture
def keys_set(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in REQUIRED_ENV:
        monkeypatch.setenv(name, "test-value")


@pytest.fixture
def postgres_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Postgres settings with no host or port override, so the defaults apply."""
    monkeypatch.setenv("POSTGRES_USER", "efe")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")
    monkeypatch.setenv("POSTGRES_DB", "news")
    monkeypatch.delenv("POSTGRES_HOST", raising=False)
    monkeypatch.delenv("POSTGRES_PORT", raising=False)


def test_passes_when_keys_set_and_db_exists(keys_set: None, db_file: Path) -> None:
    preflight(db_file)


@pytest.mark.parametrize(
    ("missing", "expected"),
    [
        ("ANTHROPIC_API_KEY", "Missing environment variables: ANTHROPIC_API_KEY"),
        ("OPENAI_API_KEY", "Missing environment variables: OPENAI_API_KEY"),
    ],
)
def test_one_missing_key_is_named(
    missing: str, expected: str, keys_set: None, db_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(missing)

    with pytest.raises(RuntimeError, match=expected):
        preflight(db_file)


def test_both_missing_keys_are_named_in_order(
    db_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError) as error:
        preflight(db_file)

    assert str(error.value) == (
        "Missing environment variables: ANTHROPIC_API_KEY, OPENAI_API_KEY, "
        "POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB"
    )


@pytest.mark.parametrize("missing", ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"])
def test_missing_postgres_setting_is_named(
    missing: str, keys_set: None, db_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(missing)

    with pytest.raises(RuntimeError, match=f"Missing environment variables: {missing}$"):
        preflight(db_file)


def test_db_uri_builds_url_with_default_host_and_port(
    postgres_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert db_uri() == "postgresql://efe:secret@localhost:5432/news"


def test_db_uri_uses_host_and_port_from_env(
    postgres_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("POSTGRES_HOST", "postgres")
    monkeypatch.setenv("POSTGRES_PORT", "6543")

    assert db_uri() == "postgresql://efe:secret@postgres:6543/news"


def test_db_uri_percent_encodes_user_and_password(
    postgres_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("POSTGRES_USER", "us er")
    monkeypatch.setenv("POSTGRES_PASSWORD", "p@ss/w:rd#1")

    # quote(..., safe="") encodes space, @, /, : and # so the URL keeps one userinfo section.
    assert db_uri() == "postgresql://us%20er:p%40ss%2Fw%3Ard%231@localhost:5432/news"


def test_empty_key_counts_as_missing(
    keys_set: None, db_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        preflight(db_file)


def test_missing_db_file_is_reported_with_its_path(keys_set: None, tmp_path: Path) -> None:
    missing = tmp_path / "absent.db"

    with pytest.raises(RuntimeError, match="News db not found") as error:
        preflight(missing)

    assert str(missing) in str(error.value)


def test_db_path_that_is_a_directory_is_rejected(keys_set: None, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="News db not found"):
        preflight(tmp_path)


def test_missing_key_is_reported_before_missing_db(
    db_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-value")
    missing_db = db_file.with_name("absent.db")

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        preflight(missing_db)
