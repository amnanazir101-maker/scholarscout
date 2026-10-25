"""Command line interface: scholarscout "your research question" -o report.md"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import anthropic
import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from scholarscout.agents import ResearchError
from scholarscout.config import Settings
from scholarscout.graph import ResearchAssistant, describe
from scholarscout.llm import LLMError


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        prog="scholarscout",
        description="Multi-agent research assistant: cited, fact-checked literature reviews.",
    )
    parser.add_argument("question", help="the research question, in quotes")
    parser.add_argument("-o", "--output", type=Path, help="save the report as a Markdown file")
    parser.add_argument("--json", type=Path, help="also save sources, fact-check and stats")
    parser.add_argument(
        "--provider", choices=["gemini", "claude"], help="LLM provider (default: gemini, free)"
    )
    parser.add_argument("--model", help="model ID (default: gemini-3.8-flash / claude-opus-5-5)")
    parser.add_argument("--max-papers", type=int, help="papers to keep after ranking")
    parser.add_argument("--full-text-papers", type=int, help="arXiv PDFs to read in full")
    parser.add_argument("--max-revisions", type=int, help="critic -> reviser rounds")
    parser.add_argument("-v", "--verbose", action="store_true", help="show warnings and logs")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # paper titles can contain any character

    settings = Settings.from_env(
        provider=args.provider,
        model=args.model,
        max_papers=args.max_papers,
        full_text_papers=args.full_text_papers,
        max_revisions=args.max_revisions,
    )
    console = Console()
    result = None
    try:
        assistant = ResearchAssistant(settings)
        with console.status("Agents at work..."):
            for agent, update in assistant.stream(args.question):
                if agent == "done":
                    result = update
                else:
                    console.print(f"[green]>[/green] {describe(agent, update)}")
    except anthropic.AuthenticationError:
        console.print("[red]Claude rejected the API key.[/red] Set ANTHROPIC_API_KEY in .env.")
        return 1
    except (anthropic.APIConnectionError, httpx.TransportError):
        console.print("[red]Could not reach the LLM API.[/red] Check the internet connection.")
        return 1
    except (LLMError, ResearchError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    console.print()
    console.print(Markdown(result.report))
    stats = result.stats
    if stats.get("cost_usd") is not None:
        cost = f", about ${stats['cost_usd']:.2f}"
    elif stats.get("provider") == "gemini":
        cost = ", Gemini free tier"
    else:
        cost = ""
    if stats.get("models_used"):
        cost += f" ({', '.join(stats['models_used'])})"
    console.print(
        f"\n[bold]{stats['papers_cited']} papers cited, "
        f"{stats['claims_supported']}/{stats['claims_checked']} statements verified, "
        f"{stats['seconds']:.0f} s{cost}.[/bold]"
    )
    if args.output:
        args.output.write_text(result.report, encoding="utf-8")
        console.print(f"Report saved to {args.output}")
    if args.json:
        args.json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        console.print(f"Details saved to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
