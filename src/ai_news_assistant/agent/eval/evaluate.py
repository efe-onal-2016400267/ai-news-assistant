import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ai_news_assistant.agent.eval.checks import ScoredTranscript, run_check
from ai_news_assistant.agent.eval.dataset import CaseDataset, Split, load_cases
from ai_news_assistant.agent.eval.report import render_report
from ai_news_assistant.agent.eval.run_config import load_run_config
from ai_news_assistant.agent.eval.transcript import Transcript

RUN_CONFIG_PATH = Path("src/ai_news_assistant/agent/eval/agent_run.toml")


def newest_run(out_dir: Path) -> Path:
    """The newest finished run; a run without `run.json` stopped before the end."""
    runs = sorted(p for p in out_dir.iterdir() if (p / "run.json").is_file())
    if not runs:
        raise ValueError(f"no finished runs in {out_dir}")
    return runs[-1]


def load_transcripts(run_dir: Path) -> list[Transcript]:
    """Load run transcripts.

    - Opens `records.jsonl` and
        does Transcript.from_dict(json.loads(line)) for each non-empty line.
    - If a line a is bad, raise an error that includes line number.
    """

    records: list[Transcript] = []
    path = run_dir / "records.jsonl"
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # skip empty lines
            try:
                records.append(Transcript.from_dict(json.loads(line)))
            # AttributeError: a line that is valid JSON but not an object (null, a list)
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
                raise ValueError(f"{path}:{line_no}: bad record ({e!r})") from e
    return records


def load_dataset_for_run(run_dir: Path) -> CaseDataset:
    """Load a dataset of cases from run_dir.

    - Reads `run.json`, takes `config.dataset` and `config.split`.
    - Calls `load_cases` for the dataset and split.
    - Uses the runs own config.
    - Compares `dataset.snapshot` with run.json["dataset"]["snapshot"]
        and raises if they differ.
    """
    with (run_dir / "run.json").open(encoding="utf-8") as f:
        meta = json.load(f)

    dataset_path = meta["config"]["dataset"]
    split: Split = meta["config"]["split"]
    run_snapshot = meta["dataset"]["snapshot"]

    dataset: CaseDataset = load_cases(path=Path(dataset_path), split=split)

    if run_snapshot != dataset.snapshot:
        raise ValueError(
            f"run used snapshot {run_snapshot}, but the cases file is for {dataset.snapshot}"
        )

    return dataset


def evaluate(run_dir: Path) -> list[ScoredTranscript]:
    """For each transcript look up the case and `call_checks`.

    - Raise if `case_id` is missing from the Dataset.
    - It's ok if some cases dont have a transcript since we can run
        a subset of the cases in run_case.py.
    """
    return score(load_transcripts(run_dir), load_dataset_for_run(run_dir))


def score(transcripts: Sequence[Transcript], dataset: CaseDataset) -> list[ScoredTranscript]:
    """Run every check on each transcript, in file order."""
    eval_results: list[ScoredTranscript] = []
    cases_by_id = {case.id: case for case in dataset.cases}
    for transcript in transcripts:
        case = cases_by_id.get(transcript.case_id)
        if case is None:
            raise ValueError(f"case {transcript.case_id!r} is not in the run's cases file")
        eval_results.append(run_check(transcript=transcript, case=case))
    return eval_results


def main(argv: Sequence[str] | None = None) -> None:
    """Take run folder as argument.

    - If no argument, take the newest folder under out_dir.
    """
    parser = argparse.ArgumentParser(description="Score a saved agent eval run.")
    parser.add_argument("run_dir", nargs="?", type=Path, help="Run folder (default: newest)")
    args = parser.parse_args(argv)

    try:
        run_dir = args.run_dir or newest_run(load_run_config(RUN_CONFIG_PATH).out_dir)
        transcripts = load_transcripts(run_dir)
        dataset = load_dataset_for_run(run_dir)
        scored = score(transcripts, dataset)
        report = render_report(run_dir.name, transcripts, dataset)
        (run_dir / "report.md").write_text(report, encoding="utf-8")
    except (ValueError, OSError) as e:  # OSError: run.json or records.jsonl missing
        raise SystemExit(f"error: {e}") from e

    for s in scored:
        print(f"{s.case_id}#{s.run_idx}: {s.outcome}")
        for r in s.results:
            if r.outcome != "pass":
                print(f"    {r.layer}/{r.id}: {r.detail}")
    print(f"Wrote {run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
