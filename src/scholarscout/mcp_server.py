"""MCP server: use ScholarScout as a tool from Claude Code, Claude Desktop or any MCP client.

scholarscout-mcp                                  # stdio transport
claude mcp add scholarscout -- scholarscout-mcp   # register it with Claude Code
"""

import logging
from functools import cache

from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

from scholarscout.config import Settings
from scholarscout.graph import ResearchAssistant
from scholarscout.papers import merge_papers, rank_papers
from scholarscout.sources import search_all

server = MCPServer(
    "scholarscout",
    instructions="Search academic papers and write cited, fact-checked literature reviews.",
    log_level="WARNING",
)
logging.getLogger("httpx").setLevel(logging.WARNING)  # one INFO line per HTTP request otherwise


@cache
def _assistant() -> ResearchAssistant:
    load_dotenv()
    return ResearchAssistant(Settings.from_env())


@server.tool()
def search_papers(query: str, max_results: int = 10) -> list[dict]:
    """Search arXiv and OpenAlex for academic papers, best match first.

    Returns title, authors, year, venue, citation count, URL and abstract for each paper.
    """
    assistant = _assistant()
    found = merge_papers(search_all(assistant.sources, [query], max_results))
    ranked = rank_papers(query, found, assistant.embedder)[:max_results]
    fields = {"title", "authors", "year", "venue", "citations", "url", "abstract"}
    return [paper.model_dump(include=fields) for paper in ranked]


@server.tool()
def literature_review(question: str) -> str:
    """Write a literature review that answers a research question.

    A team of agents plans sub-questions, searches arXiv and OpenAlex, reads the top papers,
    writes a Markdown report with numbered citations and fact-checks every cited statement.
    Takes two to five minutes.
    """
    return _assistant().run(question).report


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
