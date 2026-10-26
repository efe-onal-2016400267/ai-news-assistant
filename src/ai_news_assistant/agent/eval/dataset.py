"""Load and check the Agent Eval cases (`agent_cases.toml`).

Every `expect` check grades the last turn. Each `*_all / *_any / *_none` key is its own
check and all set checks must hold; an empty list is the same as leaving the key out.
Answer text is matched as a case-insensitive substring.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast, get_args

CaseType = Literal[
    "follow-ups-and-context-management",  # Check context management
    "filter-requests",  # Check tool arguments
    "single-turn-lookup-and-citation",  # Check citation quality
    "latest-news-recency-request",  # Check citation quality and tool arguments
    "multi-article-synthesis",  # Check correctness and citation quality
    "abstention",  # Answer not in the corpus, including general knowledge bait (James Joyce)
    "efficiency",  # Check tool call count, i/o token count, $ money cost
    "prompt-injection-inside-article",  # Canary cases
]
Split = Literal["dev", "test"]
ToolArg = Literal["sources", "days"]

CASE_TYPES = frozenset(get_args(CaseType))
SPLITS = frozenset(get_args(Split))
TOOL_ARGS = frozenset(get_args(ToolArg))
TOOL_NAMES = frozenset({"search_news", "get_article", "get_latest_articles"})

TOP_LEVEL_KEYS = frozenset({"dataset_version", "snapshot", "abstain_phrases", "case"})
CASE_REQUIRED_KEYS = frozenset({"id", "type", "split", "turns", "gold_article_ids", "expect"})
CASE_OPTIONAL_KEYS = frozenset({"setup", "precondition"})
SETUP_KEYS = frozenset({"trigger", "fixture_store", "canary"})
PRECONDITION_KEYS = frozenset({"target_not_cited_in_turn"})
EXPECT_KEYS = frozenset(
    {
        "must_abstain",
        "allow_fallback",
        "facts_all",
        "facts_any",
        "facts_none",
        "cite_all",
        "cite_any",
        "cite_none",
        "tools_all",
        "tools_any",
        "tools_none",
        "tool_args",
        "max_tool_calls",
        "grounded",
    }
)
TOOL_ARGS_KEYS = frozenset({"tool", "sources", "days", "args_absent"})


@dataclass(frozen=True)
class Checks:
    """Items (article ids or tool names) that must all, any or none appear."""

    all: tuple[str, ...] = ()
    any: tuple[str, ...] = ()
    none: tuple[str, ...] = ()


@dataclass(frozen=True)
class FactChecks:
    """Strings the final answer must contain.

    `all` holds groups: every group must match, and one wording in a group is enough.
    """

    all: tuple[tuple[str, ...], ...] = ()
    any: tuple[str, ...] = ()
    none: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolArgs:
    """Arguments the first call to `tool` must carry. At least one check is set."""

    tool: str
    sources: tuple[str, ...] | None = None  # Exact source names
    days: int | None = None
    absent: frozenset[ToolArg] = frozenset()  # Arguments that must not be passed


@dataclass(frozen=True)
class Setup:
    trigger: int | None = None  # Context management trigger, in tokens
    fixture_store: str | None = None
    canary: str | None = None


@dataclass(frozen=True)
class Precondition:
    """Checks that make the case valid; failing one means the case is wrong, not the agent."""

    target_not_cited_in_turn: int | None = None  # 1-based


@dataclass(frozen=True)
class Expect:
    must_abstain: bool = False  # Abstaining passes; stating facts fails
    allow_fallback: bool = False  # Abstaining instead of the facts is fallback, not fail
    facts: FactChecks = FactChecks()
    citations: Checks = Checks()
    tools: Checks = Checks()
    tool_args: ToolArgs | None = None
    max_tool_calls: int | None = None
    grounded: bool = False  # Stored only; graded by a judge later


@dataclass(frozen=True)
class Case:
    id: str
    type: CaseType
    split: Split
    turns: tuple[str, ...]
    gold_article_ids: tuple[str, ...]  # Empty only when must_abstain
    expect: Expect
    setup: Setup = Setup()
    precondition: Precondition = Precondition()


@dataclass(frozen=True)
class CaseDataset:
    version: int
    snapshot: str
    abstain_phrases: tuple[str, ...]
    cases: tuple[Case, ...]


def load_cases(path: Path, split: Split | None = None) -> CaseDataset:
    """Load and check the cases file, then keep only `split` (None = all).

    The whole file is checked before filtering, so rules across splits hold
    even when only one split is asked for. Raises `ValueError` on any problem.
    """
    with path.open("rb") as f:
        data = tomllib.load(f)

    unknown = set(data) - TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown top-level key(s): {', '.join(sorted(unknown))}")
    version = data.get("dataset_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError(f"{path}: dataset_version must be a positive integer")
    snapshot = data.get("snapshot")
    if not isinstance(snapshot, str) or not snapshot:
        raise ValueError(f"{path}: snapshot must be a non-empty string")
    abstain_phrases = _str_list(data.get("abstain_phrases", []), "abstain_phrases", str(path))
    if not abstain_phrases:
        raise ValueError(f"{path}: abstain_phrases must be a non-empty list")
    raw_cases = data.get("case")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"{path}: no [[case]] tables")

    cases = [parse_case(raw, abstain_phrases) for raw in raw_cases]
    _check_unique_ids(cases)
    _check_split_overlap(cases)

    if split is not None:
        cases = [c for c in cases if c.split == split]
    return CaseDataset(
        version=version, snapshot=snapshot, abstain_phrases=abstain_phrases, cases=tuple(cases)
    )


def parse_case(raw: dict[str, Any], abstain_phrases: tuple[str, ...] = ()) -> Case:
    """Turn one `[[case]]` table into a `Case`, checking its own rules."""
    cid = raw.get("id")
    if not isinstance(cid, str) or not cid:
        raise ValueError(f"case without a string id (keys: {', '.join(sorted(raw))})")

    missing = CASE_REQUIRED_KEYS - set(raw)
    if missing:
        raise ValueError(f"{cid}: missing key(s): {', '.join(sorted(missing))}")
    unknown = set(raw) - CASE_REQUIRED_KEYS - CASE_OPTIONAL_KEYS
    if unknown:
        raise ValueError(f"{cid}: unknown key(s): {', '.join(sorted(unknown))}")

    case_type = cast(CaseType, _choice(raw["type"], "type", CASE_TYPES, cid))
    turns = _str_list(raw["turns"], "turns", cid)
    if not turns:
        raise ValueError(f"{cid}: turns must be a non-empty list")
    gold = _id_list(raw["gold_article_ids"], "gold_article_ids", cid)
    setup = _setup(raw.get("setup", {}), cid)
    precondition = _precondition(raw.get("precondition", {}), len(turns), cid)
    expect = _expect(raw["expect"], cid)

    is_context = case_type == "follow-ups-and-context-management"
    if not is_context and (setup.trigger is not None or precondition != Precondition()):
        raise ValueError(f"{cid}: trigger and precondition are only for context cases")
    _check_expect(expect, gold, abstain_phrases, cid)

    return Case(
        id=cid,
        type=case_type,
        split=cast(Split, _choice(raw["split"], "split", SPLITS, cid)),
        turns=turns,
        gold_article_ids=gold,
        expect=expect,
        setup=setup,
        precondition=precondition,
    )


def _table(value: Any, name: str, keys: frozenset[str], cid: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{cid}: {name} must be a table")
    unknown = set(value) - keys
    if unknown:
        raise ValueError(f"{cid}: unknown {name} key(s): {', '.join(sorted(unknown))}")
    return value


def _choice(value: Any, key: str, allowed: frozenset[str], cid: str) -> str:
    if value not in allowed:
        raise ValueError(f"{cid}: {key} {value!r} is not one of {', '.join(sorted(allowed))}")
    return cast(str, value)


def _str_list(value: Any, key: str, cid: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise ValueError(f"{cid}: {key} must be a list of non-empty strings")
    return tuple(value)


def _id_list(value: Any, key: str, cid: str) -> tuple[str, ...]:
    """A list of non-empty strings without repeats."""
    items = _str_list(value, key, cid)
    if len(set(items)) != len(items):
        raise ValueError(f"{cid}: {key} has repeated items")
    return items


def _tool_list(value: Any, key: str, cid: str) -> tuple[str, ...]:
    items = _id_list(value, key, cid)
    unknown = set(items) - TOOL_NAMES
    if unknown:
        raise ValueError(f"{cid}: {key} has unknown tool(s): {', '.join(sorted(unknown))}")
    return items


def _int(value: Any, key: str, minimum: int, cid: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{cid}: {key} must be an integer >= {minimum}")
    return value


def _bool(value: Any, key: str, cid: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{cid}: {key} must be true or false")
    return value


def _optional_str(value: Any, key: str, cid: str) -> str | None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{cid}: {key} must be a non-empty string")
    return value


def _setup(value: Any, cid: str) -> Setup:
    table = _table(value, "setup", SETUP_KEYS, cid)
    trigger = table.get("trigger")
    return Setup(
        trigger=_int(trigger, "trigger", 1, cid) if trigger is not None else None,
        fixture_store=_optional_str(table.get("fixture_store"), "fixture_store", cid),
        canary=_optional_str(table.get("canary"), "canary", cid),
    )


def _precondition(value: Any, n_turns: int, cid: str) -> Precondition:
    table = _table(value, "precondition", PRECONDITION_KEYS, cid)
    turn = table.get("target_not_cited_in_turn")
    if turn is None:
        return Precondition()
    # The target must stay uncited before the last turn, which is the one graded
    if _int(turn, "target_not_cited_in_turn", 1, cid) >= n_turns:
        raise ValueError(f"{cid}: target_not_cited_in_turn must be before the last turn")
    return Precondition(target_not_cited_in_turn=turn)


def _fact_groups(value: Any, cid: str) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{cid}: facts_all must be a list of lists of strings")
    groups = tuple(_str_list(group, "facts_all group", cid) for group in value)
    if any(not group for group in groups):
        raise ValueError(f"{cid}: facts_all groups must not be empty")
    return groups


def _expect(value: Any, cid: str) -> Expect:
    table = _table(value, "expect", EXPECT_KEYS, cid)
    tool_args = table.get("tool_args")
    max_calls = table.get("max_tool_calls")
    return Expect(
        must_abstain=_bool(table.get("must_abstain", False), "must_abstain", cid),
        allow_fallback=_bool(table.get("allow_fallback", False), "allow_fallback", cid),
        facts=FactChecks(
            all=_fact_groups(table.get("facts_all", []), cid),
            any=_str_list(table.get("facts_any", []), "facts_any", cid),
            none=_str_list(table.get("facts_none", []), "facts_none", cid),
        ),
        citations=Checks(
            all=_id_list(table.get("cite_all", []), "cite_all", cid),
            any=_id_list(table.get("cite_any", []), "cite_any", cid),
            none=_id_list(table.get("cite_none", []), "cite_none", cid),
        ),
        tools=Checks(
            all=_tool_list(table.get("tools_all", []), "tools_all", cid),
            any=_tool_list(table.get("tools_any", []), "tools_any", cid),
            none=_tool_list(table.get("tools_none", []), "tools_none", cid),
        ),
        tool_args=_tool_args(tool_args, cid) if tool_args is not None else None,
        max_tool_calls=_int(max_calls, "max_tool_calls", 0, cid) if max_calls is not None else None,
        grounded=_bool(table.get("grounded", False), "grounded", cid),
    )


def _tool_args(value: Any, cid: str) -> ToolArgs:
    table = _table(value, "tool_args", TOOL_ARGS_KEYS, cid)
    if "tool" not in table:
        raise ValueError(f"{cid}: tool_args needs a tool")
    tool = _choice(table["tool"], "tool_args tool", TOOL_NAMES, cid)
    # An empty list is the same as leaving the key out
    sources = _id_list(table.get("sources", []), "tool_args sources", cid) or None
    days = table.get("days")
    if days is not None:
        days = _int(days, "tool_args days", 1, cid)
    absent = _id_list(table.get("args_absent", []), "args_absent", cid)
    for arg in absent:
        _choice(arg, "args_absent", TOOL_ARGS, cid)

    both = {arg for arg, set_ in (("sources", sources), ("days", days)) if set_ is not None}
    both &= set(absent)
    if both:
        raise ValueError(f"{cid}: tool_args both sets and forbids: {', '.join(sorted(both))}")
    if sources is None and days is None and not absent:
        raise ValueError(f"{cid}: tool_args must set sources, days or args_absent")
    return ToolArgs(
        tool=tool, sources=sources, days=days, absent=cast(frozenset[ToolArg], frozenset(absent))
    )


def _check_expect(
    expect: Expect, gold: tuple[str, ...], abstain_phrases: tuple[str, ...], cid: str
) -> None:
    """Rules that tie the expect checks to each other, to gold and to the abstain phrases."""
    facts, cites, tools = expect.facts, expect.citations, expect.tools
    if expect.must_abstain and expect.allow_fallback:
        raise ValueError(f"{cid}: must_abstain and allow_fallback cannot both be true")
    if expect.must_abstain != (not gold):
        raise ValueError(f"{cid}: must_abstain is true exactly when gold_article_ids is empty")
    if expect.must_abstain and (facts.all or facts.any or cites.all or cites.any):
        raise ValueError(f"{cid}: must_abstain cannot require facts or citations")
    if expect.allow_fallback and not (facts.all or facts.any):
        raise ValueError(f"{cid}: allow_fallback needs facts_all or facts_any")

    # A fact inside an abstain phrase would let an abstention pass the fact check
    phrases = [p.lower() for p in abstain_phrases]
    for fact in (*(f for group in facts.all for f in group), *facts.any):
        if any(fact.lower() in p for p in phrases):
            raise ValueError(f"{cid}: fact {fact!r} matches an abstain phrase")

    not_gold = (set(cites.all) | set(cites.any)) - set(gold)
    if not_gold:
        raise ValueError(f"{cid}: cite_all/cite_any not in gold: {', '.join(sorted(not_gold))}")
    both = set(cites.none) & set(gold)
    if both:
        raise ValueError(f"{cid}: gold article(s) also in cite_none: {', '.join(sorted(both))}")

    for (name_a, a), (name_b, b) in (
        (("tools_all", tools.all), ("tools_any", tools.any)),
        (("tools_all", tools.all), ("tools_none", tools.none)),
        (("tools_any", tools.any), ("tools_none", tools.none)),
    ):
        both = set(a) & set(b)
        if both:
            raise ValueError(f"{cid}: {name_a} and {name_b} share: {', '.join(sorted(both))}")
    if expect.tool_args is not None and expect.tool_args.tool in tools.none:
        raise ValueError(f"{cid}: tool_args tool {expect.tool_args.tool!r} is in tools_none")


def _check_unique_ids(cases: list[Case]) -> None:
    seen: set[str] = set()
    for c in cases:
        if c.id in seen:
            raise ValueError(f"{c.id}: duplicate case id")
        seen.add(c.id)


def _check_split_overlap(cases: list[Case]) -> None:
    """No article may be labeled (gold or cite_none) in both splits."""
    first_seen: dict[str, Case] = {}
    for c in cases:
        for article_id in (*c.gold_article_ids, *c.expect.citations.none):
            first = first_seen.setdefault(article_id, c)
            if first.split != c.split:
                raise ValueError(
                    f"{c.id}: article {article_id} is in {c.split}, "
                    f"but also in {first.split} ({first.id})"
                )
