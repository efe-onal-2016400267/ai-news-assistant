"""Scoring a saved run folder.

Tests named `test_bug<n>_...` describe the intended behaviour and fail on the current code on
purpose; `<n>` is the number in the bug report. Run one with `-k bug<n>`.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ai_news_assistant.agent.eval.evaluate import (
    evaluate,
    load_dataset_for_run,
    load_transcripts,
    main,
    newest_run,
)
from ai_news_assistant.agent.eval.transcript import RunError, Transcript, TurnResult

SNAPSHOT = "news-2026-10-12.db"

CASES_TOML = f"""
dataset_version = 1
snapshot = "{SNAPSHOT}"
abstain_phrases = ["I do not know"]

[[case]]
id = "one"
type = "single-turn-lookup-and-citation"
split = "dev"
turns = ["q1"]
gold_article_ids = ["a1"]
expect = {{}}

[[case]]
id = "two"
type = "single-turn-lookup-and-citation"
split = "dev"
turns = ["q2"]
gold_article_ids = ["a2"]
expect = {{}}

[[case]]
id = "held-out"
type = "single-turn-lookup-and-citation"
split = "test"
turns = ["q3"]
gold_article_ids = ["a3"]
expect = {{}}
"""

TURN = TurnResult(
    user_query="q1",
    tool_calls=(),
    retrieved_ids=("a1", "a2"),
    text="It shipped [1].",
    cited_ids=("a1",),
    grounded=True,
    structured=True,
    raw_citations=("a1",),
    raw_grounded=True,
    dropped_ids=(),
    dropped_markers=(),
    input_tokens=10,
    output_tokens=5,
    model_calls=2,
    latency_s=1.0,
)


def record(case_id: str, run_idx: int = 0, **turn_changes: object) -> Transcript:
    return Transcript(case_id=case_id, run_idx=run_idx, turns=(replace(TURN, **turn_changes),))


def write_run(
    run_dir: Path,
    cases_path: Path,
    records: list[Transcript],
    snapshot: str = SNAPSHOT,
) -> Path:
    run_dir.mkdir(parents=True)
    lines = [json.dumps(r.to_dict()) for r in records]
    (run_dir / "records.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {
        "config": {"dataset": str(cases_path), "split": "dev"},
        "dataset": {"version": 1, "snapshot": snapshot},
    }
    (run_dir / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    return run_dir


@pytest.fixture
def cases_path(tmp_path: Path) -> Path:
    path = tmp_path / "cases.toml"
    path.write_text(CASES_TOML, encoding="utf-8")
    return path


# load_transcripts


def test_load_transcripts_round_trip_skips_blank_lines(tmp_path: Path) -> None:
    ok = record("one")
    failed = Transcript("two", 1, (), RunError(turn=0, kind="KeyError", error_message="x"))
    text = json.dumps(ok.to_dict()) + "\n\n   \n" + json.dumps(failed.to_dict()) + "\n"
    (tmp_path / "records.jsonl").write_text(text, encoding="utf-8")

    assert load_transcripts(tmp_path) == [ok, failed]


def write_with_bad_third_line(run_dir: Path, bad_line: str) -> None:
    good = json.dumps(record("one").to_dict())
    (run_dir / "records.jsonl").write_text(f"{good}\n{good}\n{bad_line}\n", encoding="utf-8")


@pytest.mark.parametrize("bad_line", ["{not json", '{"case_id": "one"}', '{"turns": 5}'])
def test_load_transcripts_bad_line_names_its_number(tmp_path: Path, bad_line: str) -> None:
    write_with_bad_third_line(tmp_path, bad_line)

    with pytest.raises(ValueError, match=r"records\.jsonl:3: bad record"):
        load_transcripts(tmp_path)


@pytest.mark.parametrize("bad_line", ["null", "[1, 2]", '"text"'])
def test_bug13_non_object_line_names_its_number(tmp_path: Path, bad_line: str) -> None:
    """BUG 13: `Transcript.from_dict` calls `d.get` first, so a JSON value that is not an
    object raises AttributeError, which `load_transcripts` does not catch."""
    write_with_bad_third_line(tmp_path, bad_line)

    with pytest.raises(ValueError, match=r"records\.jsonl:3: bad record"):
        load_transcripts(tmp_path)


# load_dataset_for_run


def test_load_dataset_uses_the_runs_split(tmp_path: Path, cases_path: Path) -> None:
    run_dir = write_run(tmp_path / "run", cases_path, [])

    dataset = load_dataset_for_run(run_dir)

    assert [c.id for c in dataset.cases] == ["one", "two"]  # held-out is test split


def test_load_dataset_snapshot_mismatch_raises(tmp_path: Path, cases_path: Path) -> None:
    run_dir = write_run(tmp_path / "run", cases_path, [], snapshot="news-2020-01-01.db")

    with pytest.raises(ValueError, match="run used snapshot news-2020-01-01.db"):
        load_dataset_for_run(run_dir)


# newest_run


def test_newest_run_picks_last_folder_and_ignores_files(tmp_path: Path) -> None:
    for name in ("20261029-095151_baseline", "20261029-102647_baseline", "20261028-235959_x"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "run.json").write_text("{}", encoding="utf-8")
    (tmp_path / "zzz-notes.txt").touch()

    assert newest_run(tmp_path) == tmp_path / "20261029-102647_baseline"


def test_newest_run_empty_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no finished runs in"):
        newest_run(tmp_path)


def test_newest_run_only_unfinished_runs_raises(tmp_path: Path) -> None:
    (tmp_path / "20261029-101111_baseline").mkdir()

    with pytest.raises(ValueError, match="no finished runs in"):
        newest_run(tmp_path)


def test_bug2_newest_run_skips_a_run_that_did_not_finish(tmp_path: Path) -> None:
    """BUG 2: a crashed run (records.jsonl, no run.json) is picked as the newest."""
    finished = tmp_path / "20261029-095151_baseline"
    finished.mkdir()
    (finished / "run.json").write_text("{}", encoding="utf-8")
    crashed = tmp_path / "20261029-101111_baseline"
    crashed.mkdir()
    (crashed / "records.jsonl").touch()

    assert newest_run(tmp_path) == finished


# evaluate


def test_evaluate_scores_each_transcript_in_file_order(tmp_path: Path, cases_path: Path) -> None:
    records = [
        record("two", 0, cited_ids=("a1",)),  # cites a1, but gold of "two" is a2 -> L2 fail
        record("one", 1),
        record("one", 0, structured=False),  # L0 fail
    ]
    run_dir = write_run(tmp_path / "run", cases_path, records)

    scored = evaluate(run_dir)

    assert [(s.case_id, s.run_idx, s.outcome) for s in scored] == [
        ("two", 0, "fail"),
        ("one", 1, "pass"),
        ("one", 0, "fail"),
    ]
    # L0, L2, L3, L1
    assert [r.outcome for r in scored[0].results] == ["pass", "fail", "pass", "pass"]
    assert [r.outcome for r in scored[2].results] == ["fail", "pass", "pass", "pass"]


def test_evaluate_subset_run_is_fine(tmp_path: Path, cases_path: Path) -> None:
    run_dir = write_run(tmp_path / "run", cases_path, [record("one")])

    assert [s.case_id for s in evaluate(run_dir)] == ["one"]


def test_bug1_unknown_case_id_raises_value_error(tmp_path: Path, cases_path: Path) -> None:
    """BUG 1: `cases_by_id[...]` raises KeyError; the `is None` check after it is dead."""
    run_dir = write_run(tmp_path / "run", cases_path, [record("one"), record("deleted-case")])

    with pytest.raises(ValueError, match="deleted-case"):
        evaluate(run_dir)


# main


def test_main_prints_outcomes_and_non_pass_details(
    tmp_path: Path, cases_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = [record("one"), record("two", 2, cited_ids=("a1",))]
    run_dir = write_run(tmp_path / "run", cases_path, records)

    main([str(run_dir)])

    assert capsys.readouterr().out.splitlines() == [
        "one#0: pass",
        "two#2: fail",
        "    L2/two: Turn 1, no cited article is gold.",
        f"Wrote {run_dir / 'report.md'}",
    ]


def test_main_writes_report_md_into_the_run_folder(tmp_path: Path, cases_path: Path) -> None:
    records = [record("one"), record("two", 2, input_tokens=30)]
    run_dir = write_run(tmp_path / "run", cases_path, records)

    main([str(run_dir)])

    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert report.startswith("# Agent eval: `run`\n")
    assert "## Outcomes" in report
    # "two" has an L2 fail (cited a1, gold is a2), so its one run is a fail
    assert "| two | 1 | 0 | 0 | 1 | 0 |" in report
    assert "- `two#2` `L2`: Turn 1, no cited article is gold." in report
    assert "## Efficiency" in report
    # input tokens [10, 30] -> median 20, max 30
    assert "| single-turn-lookup-and-citation | 2 | 0 / 0 | 20 / 30 | 5 / 5 | 1.0 / 1.0 |" in report


def test_main_snapshot_mismatch_exits(tmp_path: Path, cases_path: Path) -> None:
    run_dir = write_run(tmp_path / "run", cases_path, [], snapshot="other.db")

    with pytest.raises(SystemExit, match="^error: run used snapshot other.db"):
        main([str(run_dir)])
    assert not (run_dir / "report.md").exists()


def test_bug1_main_unknown_case_id_exits_with_message(tmp_path: Path, cases_path: Path) -> None:
    """BUG 1: main catches only ValueError, so the KeyError ends in a traceback."""
    run_dir = write_run(tmp_path / "run", cases_path, [record("deleted-case")])

    with pytest.raises(SystemExit, match="^error: "):
        main([str(run_dir)])


@pytest.mark.parametrize("missing", ["run.json", "records.jsonl"])
def test_bug2_main_missing_run_file_exits_with_message(
    tmp_path: Path, cases_path: Path, missing: str
) -> None:
    """BUG 2: FileNotFoundError is not caught by main."""
    run_dir = write_run(tmp_path / "run", cases_path, [record("one")])
    (run_dir / missing).unlink()

    with pytest.raises(SystemExit, match="^error: "):
        main([str(run_dir)])
