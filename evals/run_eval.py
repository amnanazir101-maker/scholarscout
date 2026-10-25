"""Compare ScholarScout with a single-agent RAG baseline on a fixed set of research questions.

    python evals/run_eval.py --limit 3                              # quick run on 3 questions
    python evals/run_eval.py --judge-model gemini-3.5-flash-lite    # judge with another model

Both systems get the same sources, ranking, PDF reading and evidence budget. A fresh judge
(not the critic inside the pipeline) then fact-checks every report the same way. Results are
saved to evals/results/<timestamp>/ (one JSON per question and summary.md).

A question takes about 25 LLM calls. On Gemini's free tier, daily request limits apply, so
start with --limit; on a paid API, check the cost of a --limit 1 run first.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from scholarscout.agents import fact_check
from scholarscout.baseline import run_baseline
from scholarscout.config import Settings
from scholarscout.graph import ResearchAssistant
from scholarscout.llm import LLM, Usage, make_llm
from scholarscout.schemas import Review

HERE = Path(__file__).parent
SYSTEMS = {"scholarscout": "ScholarScout (multi-agent)", "baseline": "Single-agent RAG"}
METRICS = {
    "citation_support": ("Cited statements supported by their source", "{:.0%}"),
    "citation_coverage": ("Statements that carry a citation", "{:.0%}"),
    "invalid_citations": ("Citations to papers that do not exist", "{:.2f}"),
    "papers_cited": ("Distinct papers cited", "{:.1f}"),
    "seconds": ("Time per report (s)", "{:.0f}"),
    "cost_usd": ("Claude cost per report (USD)", "${:.2f}"),
}


def score(review: Review, seconds: float, usage: Usage, model: str) -> dict:
    cost = usage.cost(model)
    valid = {pid for claim in review.claims for pid in claim.citations} - set(
        review.invalid_citations
    )
    return {
        "citation_support": round(review.support_rate, 3),
        "citation_coverage": round(review.citation_coverage, 3),
        "invalid_citations": len(review.invalid_citations),
        "papers_cited": len(valid),
        "claims": len(review.claims),
        "seconds": round(seconds, 1),
        "cost_usd": round(cost, 4) if cost is not None else None,
    }


def summarize(rows: list[dict], model: str, judge_model: str) -> str:
    done = [row for row in rows if "error" not in row]
    lines = [
        "# ScholarScout evaluation",
        "",
        f"- Date: {datetime.now():%Y-%m-%d}",
        f"- Questions: {len(done)} of {len(rows)} completed",
        f"- Model: `{model}`, judge: `{judge_model}`",
        "",
        "| Metric | " + " | ".join(SYSTEMS.values()) + " |",
        "|---|" + "---|" * len(SYSTEMS),
    ]
    for key, (label, fmt) in METRICS.items():
        cells = []
        for system in SYSTEMS:
            values = [row[system][key] for row in done if row[system][key] is not None]
            cells.append(fmt.format(statistics.mean(values)) if values else "n/a")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def evaluate(
    questions: list[dict], assistant: ResearchAssistant, judge: LLM, run_dir: Path
) -> list[dict]:
    """Run both systems on every question; save one JSON file per question in run_dir."""
    llm, settings = assistant.llm, assistant.settings
    rows = []
    for n, item in enumerate(questions, 1):
        question = item["question"]
        print(f"[{n}/{len(questions)}] {question}")
        row: dict = {"id": item["id"], "question": question}
        try:
            team = assistant.new_team()
            before, started = llm.usage.snapshot(), time.perf_counter()
            result = assistant.run(question, team=team)
            seconds, usage = time.perf_counter() - started, llm.usage.snapshot() - before
            review = fact_check(result.draft, team.registry, team.store, judge)
            row["scholarscout"] = score(review, seconds, usage, llm.model)

            before = llm.usage.snapshot()
            base = run_baseline(
                question, settings, llm, assistant.sources, assistant.embedder, assistant.reader
            )
            base_review = fact_check(base.draft, base.registry, base.store, judge)
            row["baseline"] = score(
                base_review, base.seconds, llm.usage.snapshot() - before, llm.model
            )
            details = {
                "scholarscout_report": result.report,
                "scholarscout_review": review.model_dump(),
                "baseline_draft": base.draft,
                "baseline_review": base_review.model_dump(),
            }
            print(
                f"    citation support: ScholarScout {row['scholarscout']['citation_support']:.0%}"
                f", baseline {row['baseline']['citation_support']:.0%}"
            )
        except Exception as exc:  # keep going; the summary reports how many finished
            row["error"] = f"{type(exc).__name__}: {exc}"
            details = {}
            print(f"    failed: {row['error']}")
        rows.append(row)
        (run_dir / f"{n:02d}-{item['id']}.json").write_text(
            json.dumps({**row, **details}, indent=2), encoding="utf-8"
        )
    return rows


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--questions", type=Path, default=HERE / "questions.jsonl")
    parser.add_argument("--limit", type=int, help="only run the first N questions")
    parser.add_argument("--judge-model", help="model for the fact-check (default: same model)")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()

    settings = Settings.from_env()
    assistant = ResearchAssistant(settings)
    judge = make_llm(settings, model=args.judge_model)
    lines = args.questions.read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line) for line in lines if line.strip()][: args.limit]

    run_dir = args.out / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = evaluate(questions, assistant, judge, run_dir)
    summary = summarize(rows, assistant.llm.model, judge.model)
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
