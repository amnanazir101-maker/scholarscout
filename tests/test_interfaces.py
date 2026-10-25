"""Smoke tests for the CLI, the MCP server tools and the Streamlit app."""

from pathlib import Path

import pytest

from scholarscout import cli
from tests.fakes import FakeLLM
from tests.test_pipeline import QUESTION, make_assistant

ROOT = Path(__file__).parents[1]


def test_cli_saves_report_and_details(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ResearchAssistant", lambda settings: make_assistant(FakeLLM()))
    report, details = tmp_path / "review.md", tmp_path / "review.json"

    assert cli.main([QUESTION, "-o", str(report), "--json", str(details)]) == 0
    assert "## References" in report.read_text(encoding="utf-8")
    assert '"support_rate": 1.0' in details.read_text(encoding="utf-8")


def test_mcp_tools_use_the_assistant(monkeypatch):
    mcp_server = pytest.importorskip("scholarscout.mcp_server")
    monkeypatch.setattr(mcp_server, "_assistant", lambda: make_assistant(FakeLLM()))

    papers = mcp_server.search_papers("benchmark for efficient transformers", max_results=2)
    assert len(papers) == 2
    assert set(papers[0]) == {"title", "authors", "year", "venue", "citations", "url", "abstract"}
    assert "## References" in mcp_server.literature_review(QUESTION)


def test_streamlit_app_renders():
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
    assert not app.exception
    assert app.title[0].value.endswith("ScholarScout")

    app.button[0].click().run(timeout=60)  # an example question fills the text box
    assert app.text_area[0].value.startswith("How do state space models")
    assert app.button[-1].label == "Start research"
