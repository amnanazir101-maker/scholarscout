from scholarscout.baseline import run_baseline
from scholarscout.config import Settings
from scholarscout.graph import ResearchAssistant, describe
from tests.fakes import FOLLOW_UP_PAPER, FakeEmbedder, FakeLLM, FakeReader, FakeSource

QUESTION = "How do state space models compare with transformers?"


def make_assistant(llm: FakeLLM, source: FakeSource | None = None, **settings) -> ResearchAssistant:
    return ResearchAssistant(
        Settings(**settings),
        llm=llm,
        sources=[source or FakeSource()],
        embedder=FakeEmbedder(),
        reader=FakeReader(),
    )


def test_full_workflow_writes_a_checked_report():
    llm = FakeLLM()
    result = make_assistant(llm).run(QUESTION)

    # planner, 2 parallel researchers, writer, critic, reviser, critic again
    assert llm.calls.count("ResearchPlan") == 1
    assert llm.calls.count("ResearchNote") == 2
    assert llm.calls.count("writer") == 1
    assert llm.calls.count("reviser") == 1
    assert llm.calls.count("ClaimVerdicts") == 2
    assert result.stats["revisions"] == 1

    # the revised draft passes: no unsupported claims, no invalid IDs, numbered references
    assert result.review.passed
    assert result.stats["support_rate"] == 1.0
    assert "[P" not in result.report
    assert "Selective state space models scale linearly with sequence length [1]." in result.report
    assert "## References\n1. Ada Lovelace and Alan Turing" in result.report
    assert all(paper.title in result.report for paper in result.cited)
    assert "Fact-check notes" not in result.report
    assert [paper.id for paper in result.cited] == ["P1", "P2"]


def test_unfixed_claims_are_reported_when_revisions_run_out():
    result = make_assistant(FakeLLM(), max_revisions=0).run(QUESTION)

    assert result.stats["revisions"] == 0
    assert result.review.invalid_citations == ["P99"]
    assert "P99" not in result.report and "[]" not in result.report
    assert "## Fact-check notes" in result.report
    assert "This sentence is UNSUPPORTED by its source" in result.report


def test_researcher_runs_a_follow_up_search_when_evidence_is_thin():
    source = FakeSource(follow_up=FOLLOW_UP_PAPER)
    result = make_assistant(FakeLLM(thin_evidence=True), source).run(QUESTION)

    assert "selective scan kernels" in source.queries
    assert all(f.follow_up_query == "selective scan kernels" for f in result.findings)
    assert FOLLOW_UP_PAPER.title in [paper.title for paper in result.papers]


def test_stream_reports_every_agent():
    events = [agent for agent, _ in make_assistant(FakeLLM()).stream(QUESTION)]
    assert events[:3] == ["planner", "searcher", "reader"]
    assert events.count("researcher") == 2
    assert events[-3:] == ["critic", "finalizer", "done"]


def test_describe_is_readable():
    assistant = make_assistant(FakeLLM())
    lines = [
        describe(agent, update) for agent, update in assistant.stream(QUESTION) if agent != "done"
    ]
    assert lines[0] == "Planner split the question into 2 sub-questions."
    assert any(line.startswith("Critic confirmed") for line in lines)


def test_baseline_makes_one_writing_call():
    llm = FakeLLM()
    run = run_baseline(QUESTION, Settings(), llm, [FakeSource()], FakeEmbedder(), FakeReader())
    assert llm.calls == ["baseline"]
    assert "[P1]" in run.draft
    assert len(run.registry) == 3
