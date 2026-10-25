from evals.run_eval import evaluate, summarize
from tests.fakes import FakeLLM
from tests.test_pipeline import QUESTION, make_assistant


class BrokenLLM(FakeLLM):
    def structured(self, *args, **kwargs):
        raise RuntimeError("API is down")


def test_evaluation_scores_both_systems(tmp_path):
    questions = [{"id": "ssm", "question": QUESTION}]
    [row] = evaluate(questions, make_assistant(FakeLLM()), FakeLLM(), tmp_path)

    assert row["scholarscout"]["citation_support"] == 1.0
    assert row["scholarscout"]["invalid_citations"] == 0
    assert row["baseline"]["papers_cited"] == 1
    assert (tmp_path / "01-ssm.json").exists()

    summary = summarize([row], "fake-model", "fake-model")
    assert "| Cited statements supported by their source | 100% | 100% |" in summary
    assert "Questions: 1 of 1 completed" in summary


def test_failed_questions_are_skipped_in_the_summary(tmp_path):
    questions = [{"id": "q", "question": QUESTION}]
    rows = evaluate(questions, make_assistant(BrokenLLM()), FakeLLM(), tmp_path)
    assert rows[0]["error"] == "RuntimeError: API is down"
    assert "Questions: 0 of 1 completed" in summarize(rows, "m", "m")
