import json
from pathlib import Path
from typing import Any

import pytest

from ai_news_assistant.agent.eval import dataset
from ai_news_assistant.agent.eval.dataset import (
    Checks,
    FactChecks,
    Precondition,
    Setup,
    load_cases,
    parse_case,
)

REAL_FILE = Path(dataset.__file__).with_name("agent_cases.toml")
CTX = "follow-ups-and-context-management"


def case(
    cid: str = "c-1",
    *,
    type_: str = "single-turn-lookup-and-citation",
    split: str = "dev",
    turns: tuple[str, ...] = ("What happened?",),
    gold: tuple[str, ...] = ("a1",),
    expect: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """A minimal valid case as a TOML table; `extra` adds or replaces case keys."""
    raw: dict[str, Any] = {
        "id": cid,
        "type": type_,
        "split": split,
        "turns": list(turns),
        "gold_article_ids": list(gold),
        "expect": dict(expect or {}),
    }
    raw.update(extra)
    return raw


def abstain_case(cid: str = "c-abstain", **expect: Any) -> dict[str, Any]:
    return case(cid, gold=(), expect={"must_abstain": True, **expect})


def _table_lines(path: str, table: dict[str, Any], array: bool = False) -> list[str]:
    """TOML lines for one table; dict values become sub-tables after the plain keys."""
    lines = [f"[[{path}]]" if array else f"[{path}]"]
    lines += [f"{k} = {json.dumps(v)}" for k, v in table.items() if not isinstance(v, dict)]
    for k, v in table.items():
        if isinstance(v, dict):
            lines += _table_lines(f"{path}.{k}", v)
    return lines


def write_toml(path: Path, cases: list[dict[str, Any]], **top: Any) -> Path:
    """Write cases as TOML. A top-level key set to None is left out.

    JSON strings, numbers, booleans and lists are valid TOML values.
    """
    values = {"dataset_version": 1, "snapshot": "snap.db", "abstain_phrases": ["don't have"]}
    values.update(top)
    lines = [f"{k} = {json.dumps(v)}" for k, v in values.items() if v is not None]
    for c in cases:
        lines += _table_lines("case", c, array=True)
    file = path / "cases.toml"
    file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return file


# Real file


def test_real_file_loads() -> None:
    data = load_cases(REAL_FILE)
    by_id = {c.id: c for c in data.cases}

    assert data.version == 1
    assert data.snapshot == "news-2026-10-12.db"
    assert data.abstain_phrases == (
        "not in",
        "isn't in",
        "don't have",
        "do not have",
        "not available",
        "don't know",
    )
    assert len(data.cases) == 9
    assert by_id["ctx-unmentioned-follow-up"].precondition.target_not_cited_in_turn == 1
    assert by_id["filter-none-for-plain-topic"].expect.tool_args is not None
    assert by_id["filter-none-for-plain-topic"].expect.tool_args.absent == {"sources", "days"}
    assert by_id["filter-unknown-outlet"].expect.must_abstain
    assert by_id["filter-unknown-outlet"].gold_article_ids == ()


@pytest.mark.parametrize(("split", "count"), [("dev", 9), ("test", 0)])
def test_real_file_split(split: Any, count: int) -> None:
    assert len(load_cases(REAL_FILE, split=split).cases) == count


# Happy path


def test_split_filter_keeps_only_that_split(tmp_path: Path) -> None:
    file = write_toml(tmp_path, [case("d", gold=("a1",)), case("t", split="test", gold=("a2",))])

    assert [c.id for c in load_cases(file).cases] == ["d", "t"]
    assert [c.id for c in load_cases(file, split="dev").cases] == ["d"]
    assert [c.id for c in load_cases(file, split="test").cases] == ["t"]


def test_minimal_case_gets_defaults() -> None:
    c = parse_case(case())

    assert c.turns == ("What happened?",)
    assert c.gold_article_ids == ("a1",)
    assert c.setup == Setup()
    assert c.precondition == Precondition()
    assert c.expect.facts == FactChecks()
    assert c.expect.citations == Checks()
    assert c.expect.tools == Checks()
    assert c.expect.tool_args is None
    assert c.expect.max_tool_calls is None
    assert not c.expect.must_abstain
    assert not c.expect.allow_fallback
    assert not c.expect.grounded


def test_facts_all_keeps_groups_in_order() -> None:
    c = parse_case(case(expect={"facts_all": [["40,363", "40363"], ["September"]]}))

    assert c.expect.facts.all == (("40,363", "40363"), ("September",))


def test_empty_lists_are_the_same_as_missing_keys() -> None:
    empty = {
        "facts_all": [],
        "facts_any": [],
        "facts_none": [],
        "cite_all": [],
        "cite_any": [],
        "cite_none": [],
        "tools_all": [],
        "tools_any": [],
        "tools_none": [],
    }
    assert parse_case(case(expect=empty)) == parse_case(case())

    with_empty_sources = parse_case(
        case(expect={"tool_args": {"tool": "search_news", "sources": [], "days": 3}})
    )
    assert with_empty_sources.expect.tool_args is not None
    assert with_empty_sources.expect.tool_args.sources is None
    assert with_empty_sources.expect.tool_args.days == 3


def test_full_context_case_parses() -> None:
    c = parse_case(
        case(
            type_=CTX,
            turns=("first", "second"),
            setup={"trigger": 3000},
            precondition={"target_not_cited_in_turn": 1},
            expect={
                "allow_fallback": True,
                "facts_any": ["40,363"],
                "cite_any": ["a1"],
                "tools_all": ["search_news"],
                "tools_none": ["get_latest_articles"],
                "max_tool_calls": 0,
                "tool_args": {"tool": "search_news", "sources": ["The Verge"]},
            },
        ),
        ("don't have",),
    )

    assert c.setup.trigger == 3000
    assert c.precondition.target_not_cited_in_turn == 1
    assert c.expect.tools == Checks(all=("search_news",), none=("get_latest_articles",))
    assert c.expect.max_tool_calls == 0
    assert c.expect.tool_args is not None
    assert c.expect.tool_args.sources == ("The Verge",)


def test_abstain_case_parses() -> None:
    c = parse_case(abstain_case(facts_none=["Reuters reported"]))

    assert c.expect.must_abstain
    assert c.expect.facts.none == ("Reuters reported",)


# File-level errors


@pytest.mark.parametrize(
    ("top", "match"),
    [
        ({"extra": 1}, "unknown top-level key"),
        ({"dataset_version": None}, "dataset_version must be a positive integer"),
        ({"dataset_version": 0}, "dataset_version must be a positive integer"),
        ({"dataset_version": True}, "dataset_version must be a positive integer"),
        ({"snapshot": None}, "snapshot must be a non-empty string"),
        ({"snapshot": ""}, "snapshot must be a non-empty string"),
        ({"abstain_phrases": None}, "abstain_phrases must be a non-empty list"),
        ({"abstain_phrases": []}, "abstain_phrases must be a non-empty list"),
        ({"abstain_phrases": [""]}, "abstain_phrases must be a list of non-empty strings"),
    ],
)
def test_bad_top_level_raises(tmp_path: Path, top: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        load_cases(write_toml(tmp_path, [case()], **top))


def test_no_cases_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"no \[\[case\]\] tables"):
        load_cases(write_toml(tmp_path, []))


def test_duplicate_id_raises(tmp_path: Path) -> None:
    file = write_toml(tmp_path, [case("same", gold=("a1",)), case("same", gold=("a2",))])
    with pytest.raises(ValueError, match="same: duplicate case id"):
        load_cases(file)


def test_gold_in_both_splits_raises(tmp_path: Path) -> None:
    file = write_toml(tmp_path, [case("d", gold=("a1",)), case("t", split="test", gold=("a1",))])
    with pytest.raises(ValueError, match=r"t: article a1 is in test, but also in dev \(d\)"):
        load_cases(file)


def test_cite_none_in_other_split_raises(tmp_path: Path) -> None:
    cases = [
        case("d", gold=("a1",), expect={"cite_none": ["a2"]}),
        case("t", split="test", gold=("a2",)),
    ]
    with pytest.raises(ValueError, match="article a2 is in test, but also in dev"):
        load_cases(write_toml(tmp_path, cases))


def test_split_overlap_is_checked_before_filtering(tmp_path: Path) -> None:
    file = write_toml(tmp_path, [case("d", gold=("a1",)), case("t", split="test", gold=("a1",))])
    with pytest.raises(ValueError, match="also in dev"):
        load_cases(file, split="dev")


# Case-level errors


BAD_CASES: list[tuple[dict[str, Any], str]] = [
    # Keys
    ({k: v for k, v in case().items() if k != "id"}, "case without a string id"),
    ({k: v for k, v in case().items() if k != "turns"}, r"missing key\(s\): turns"),
    (case(extra=1), r"unknown key\(s\): extra"),
    (case(type_=CTX, setup={"foo": 1}), r"unknown setup key\(s\): foo"),
    (case(type_=CTX, precondition={"foo": 1}), r"unknown precondition key\(s\): foo"),
    (case(expect={"trigger": 3000}), r"unknown expect key\(s\): trigger"),
    (
        case(expect={"tool_args": {"tool": "search_news", "days": 1, "k": 5}}),
        r"unknown tool_args key\(s\): k",
    ),
    (case(setup=[1]), "setup must be a table"),
    # Values
    (case(type_="context"), "type 'context' is not one of"),
    (case(split="train"), "split 'train' is not one of"),
    (case(turns=()), "turns must be a non-empty list"),
    (case(turns=(" ",)), "turns must be a list of non-empty strings"),
    (case(gold=("a1", "a1")), "gold_article_ids has repeated items"),
    (case(expect={"tools_any": ["compare_sources"]}), r"unknown tool\(s\): compare_sources"),
    (case(type_=CTX, setup={"trigger": 0}), "trigger must be an integer >= 1"),
    (case(expect={"max_tool_calls": -1}), "max_tool_calls must be an integer >= 0"),
    (case(expect={"max_tool_calls": True}), "max_tool_calls must be an integer"),
    (
        case(expect={"tool_args": {"tool": "search_news", "days": 0}}),
        "tool_args days must be an integer >= 1",
    ),
    (case(expect={"grounded": "yes"}), "grounded must be true or false"),
    (case(type_=CTX, setup={"canary": ""}), "canary must be a non-empty string"),
    # Precondition and context-only keys
    (
        case(type_=CTX, turns=("a", "b"), precondition={"target_not_cited_in_turn": 2}),
        "target_not_cited_in_turn must be before the last turn",
    ),
    (
        case(type_=CTX, turns=("a", "b"), precondition={"target_not_cited_in_turn": 0}),
        "target_not_cited_in_turn must be an integer >= 1",
    ),
    (case(setup={"trigger": 3000}), "only for context cases"),
    (
        case(turns=("a", "b"), precondition={"target_not_cited_in_turn": 1}),
        "only for context cases",
    ),
    # facts_all groups
    (case(expect={"facts_all": "40,363"}), "facts_all must be a list of lists"),
    (case(expect={"facts_all": [[]]}), "facts_all groups must not be empty"),
    (case(expect={"facts_all": [[1]]}), "facts_all group must be a list of non-empty strings"),
    # Abstain and fallback
    (abstain_case(allow_fallback=True), "must_abstain and allow_fallback cannot both be true"),
    (case(expect={"must_abstain": True}), "must_abstain is true exactly when"),
    (case(gold=()), "must_abstain is true exactly when"),
    (abstain_case(facts_any=["x"]), "must_abstain cannot require facts or citations"),
    (abstain_case(cite_any=["a1"]), "must_abstain cannot require facts or citations"),
    (case(expect={"allow_fallback": True}), "allow_fallback needs facts_all or facts_any"),
    (
        case(expect={"allow_fallback": True, "facts_none": ["x"]}),
        "allow_fallback needs facts_all or facts_any",
    ),
    # Citations
    (case(expect={"cite_any": ["a2"]}), "cite_all/cite_any not in gold: a2"),
    (case(expect={"cite_all": ["a2"]}), "cite_all/cite_any not in gold: a2"),
    (case(expect={"cite_none": ["a1"]}), "gold article.* also in cite_none: a1"),
    (case(expect={"cite_none": ["a2", "a2"]}), "cite_none has repeated items"),
    # Tools
    (
        case(expect={"tools_all": ["search_news"], "tools_any": ["search_news"]}),
        "tools_all and tools_any share: search_news",
    ),
    (
        case(expect={"tools_all": ["search_news"], "tools_none": ["search_news"]}),
        "tools_all and tools_none share: search_news",
    ),
    (
        case(expect={"tools_any": ["get_article"], "tools_none": ["get_article"]}),
        "tools_any and tools_none share: get_article",
    ),
    (
        case(
            expect={"tools_none": ["search_news"], "tool_args": {"tool": "search_news", "days": 1}}
        ),
        "tool_args tool 'search_news' is in tools_none",
    ),
    # tool_args
    (case(expect={"tool_args": {"days": 1}}), "tool_args needs a tool"),
    (
        case(expect={"tool_args": {"tool": "compare_sources", "days": 1}}),
        "tool_args tool 'compare_sources' is not one of",
    ),
    (
        case(expect={"tool_args": {"tool": "search_news", "args_absent": ["k"]}}),
        "args_absent 'k' is not one of",
    ),
    (
        case(
            expect={
                "tool_args": {"tool": "search_news", "sources": ["X"], "args_absent": ["sources"]}
            }
        ),
        "tool_args both sets and forbids: sources",
    ),
    (
        case(expect={"tool_args": {"tool": "search_news"}}),
        "tool_args must set sources, days or args_absent",
    ),
]


@pytest.mark.parametrize(("raw", "match"), BAD_CASES)
def test_bad_case_raises(raw: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_case(raw)


def test_case_errors_name_the_case_id() -> None:
    with pytest.raises(ValueError, match="^my-case: split 'train'"):
        parse_case(case("my-case", split="train"))


# Facts against abstain phrases


@pytest.mark.parametrize(
    "expect",
    [
        {"facts_any": ["have"]},  # "have" is inside "don't have": any abstention would pass
        {"facts_any": ["DON'T HAVE"]},  # Case is ignored
        {"facts_all": [["40,363"], ["know", "40363"]]},  # Checked inside every group
    ],
)
def test_fact_inside_abstain_phrase_raises(expect: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="matches an abstain phrase"):
        parse_case(case(expect=expect), ("don't have", "don't know"))


def test_fact_longer_than_abstain_phrase_is_allowed() -> None:
    # An abstention says "don't have", not this whole fact, so it cannot pass the check
    c = parse_case(case(expect={"facts_any": ["we don't have data on X"]}), ("don't have",))

    assert c.expect.facts.any == ("we don't have data on X",)


def test_facts_none_may_match_abstain_phrase() -> None:
    c = parse_case(case(expect={"facts_none": ["don't have"]}), ("don't have",))

    assert c.expect.facts.none == ("don't have",)
