"""The agents. Each public method of `ResearchTeam` is one node of the LangGraph workflow."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from langgraph.types import Send

from scholarscout import prompts
from scholarscout.citations import (
    cited_ids,
    extract_claims,
    normalize_citations,
    number_citations,
    reference_list,
    remove_citations,
)
from scholarscout.config import Settings
from scholarscout.llm import LLM
from scholarscout.papers import PaperRegistry, id_number, merge_papers, rank_papers
from scholarscout.reader import FullTextReader
from scholarscout.schemas import (
    CheckedClaim,
    Claim,
    ClaimVerdict,
    ClaimVerdicts,
    Finding,
    Paper,
    ResearchNote,
    ResearchPlan,
    Review,
    SubQuestion,
)
from scholarscout.sources import PaperSource, search_all
from scholarscout.state import ResearchState, ResearchTask
from scholarscout.store import Chunk, PaperStore

MAX_SUB_QUESTIONS = 4
FOLLOW_UP_PAPERS = 4  # papers one follow-up search may add
CLAIMS_PER_CHECK = 12  # claims judged per critic call
PASSAGES_PER_PAPER = 6  # most full-text passages the critic sees per cited paper


class ResearchError(RuntimeError):
    """The run cannot continue, for example because no papers were found."""


def format_excerpts(chunks: list[Chunk], registry: PaperRegistry) -> str:
    grouped: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.paper_id, []).append(chunk)
    blocks = []
    for paper_id, items in grouped.items():
        paper = registry.get(paper_id)
        lines = [f"[{paper_id}] {paper.title} ({paper.author_label()}, {paper.year or 'n.d.'})"]
        lines += [
            f"- {'Abstract' if chunk.kind == 'abstract' else 'Passage'}: {chunk.text}"
            for chunk in items
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def format_notes(findings: list[Finding]) -> str:
    return "\n\n".join(f"### {finding.sub_question}\n{finding.text}" for finding in findings)


def format_catalog(papers: list[Paper]) -> str:
    return "\n".join(
        f"[{paper.id}] {paper.title} ({paper.author_label()}, {paper.year or 'n.d.'})"
        for paper in papers
    )


def ordered_findings(state: ResearchState) -> list[Finding]:
    """Findings in the planner's order (parallel researchers finish in any order)."""
    order = {sq.question: i for i, sq in enumerate(state.get("plan", []))}
    return sorted(state.get("findings", []), key=lambda f: order.get(f.sub_question, len(order)))


def fact_check(report: str, registry: PaperRegistry, store: PaperStore, llm: LLM) -> Review:
    """Check every cited statement of a report against the papers it cites."""
    claims, cited, total = extract_claims(report)
    known = registry.ids()
    invalid = sorted(
        {pid for claim in claims for pid in claim.citations if pid not in known}, key=id_number
    )
    checkable = [
        claim.model_copy(update={"citations": [pid for pid in claim.citations if pid in known]})
        for claim in claims
        if any(pid in known for pid in claim.citations)
    ]
    batches = [
        checkable[i : i + CLAIMS_PER_CHECK] for i in range(0, len(checkable), CLAIMS_PER_CHECK)
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = pool.map(lambda batch: _judge(batch, registry, store, llm), batches)
        verdicts = {verdict.claim_id: verdict for batch in results for verdict in batch}

    checked = []
    for claim in claims:
        verdict = verdicts.get(claim.id)
        if verdict is not None:
            supported, reason = verdict.supported, verdict.reason
        elif not any(pid in known for pid in claim.citations):
            supported, reason = False, "Cites only papers that do not exist."
        else:
            supported, reason = False, "The fact-checker returned no verdict."
        checked.append(CheckedClaim(**claim.model_dump(), supported=supported, reason=reason))
    return Review(
        claims=checked, invalid_citations=invalid, cited_statements=cited, total_statements=total
    )


def _judge(
    claims: list[Claim], registry: PaperRegistry, store: PaperStore, llm: LLM
) -> list[ClaimVerdict]:
    """One critic call: the claims plus the abstract and best passages of each cited paper."""
    cited_by: dict[str, list[Claim]] = {}
    for claim in claims:
        for pid in claim.citations:
            cited_by.setdefault(pid, []).append(claim)

    blocks = []
    for pid in sorted(cited_by, key=id_number):
        paper = registry.get(pid)
        hits = [
            store.search(claim.text, k=2, paper_ids=[pid], kind="full_text")
            for claim in cited_by[pid]
        ]
        passages: list[str] = []
        for rank in range(2):  # every claim's best passage first, then the runners-up
            for chunks in hits:
                if rank < len(chunks) and chunks[rank].text not in passages:
                    passages.append(chunks[rank].text)
        lines = [f"[{pid}] {paper.title} ({paper.year or 'n.d.'})", f"Abstract: {paper.abstract}"]
        lines += [f"Passage: {text}" for text in passages[:PASSAGES_PER_PAPER]]
        blocks.append("\n".join(lines))

    claim_lines = "\n".join(
        f"{claim.id}. {claim.text} (cites: {', '.join(claim.citations)})" for claim in claims
    )
    prompt = prompts.CRITIC_PROMPT.format(evidence="\n\n".join(blocks), claims=claim_lines)
    result = llm.structured(prompts.CRITIC_SYSTEM, prompt, ClaimVerdicts)
    wanted = {claim.id for claim in claims}
    return [verdict for verdict in result.verdicts if verdict.claim_id in wanted]


class ResearchTeam:
    """The agents of one research run, sharing the run's paper registry and vector store."""

    def __init__(
        self,
        settings: Settings,
        llm: LLM,
        sources: Sequence[PaperSource],
        store: PaperStore,
        reader: FullTextReader,
        registry: PaperRegistry | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.sources = sources
        self.store = store
        self.reader = reader
        self.registry = registry or PaperRegistry()

    # Planner: question -> sub-questions with search queries ---------------------------
    def plan(self, state: ResearchState) -> dict:
        plan = self.llm.structured(
            prompts.PLANNER_SYSTEM,
            prompts.PLANNER_PROMPT.format(question=state["question"]),
            ResearchPlan,
            effort="low",
        )
        sub_questions = [
            sq for sq in plan.sub_questions if sq.question.strip() and sq.search_query.strip()
        ][:MAX_SUB_QUESTIONS]
        if not sub_questions:
            question = state["question"]
            sub_questions = [SubQuestion(question=question, search_query=question)]
        return {"plan": sub_questions}

    # Searcher: query arXiv + OpenAlex, merge duplicates, keep the best papers ----------
    def search(self, state: ResearchState) -> dict:
        question = state["question"]
        queries = list(dict.fromkeys([sq.search_query for sq in state["plan"]] + [question]))
        found = search_all(self.sources, queries, self.settings.results_per_query)
        candidates = merge_papers(found)
        if not candidates:
            raise ResearchError(
                "No papers found. Check the internet connection or rephrase the question."
            )
        ranked = rank_papers(question, candidates, self.store.embedder)
        selected = [self.registry.add(paper)[0] for paper in ranked[: self.settings.max_papers]]
        return {
            "papers": {paper.id: paper for paper in selected},
            "stats": {"queries": len(queries), "candidates": len(candidates)},
        }

    # Reader: download the top arXiv PDFs, index every paper in the vector store --------
    def read(self, state: ResearchState) -> dict:
        papers = sorted(state["papers"].values(), key=lambda p: id_number(p.id))
        to_read = [paper for paper in papers if paper.arxiv_id][: self.settings.full_text_papers]
        with ThreadPoolExecutor(max_workers=3) as pool:
            texts = dict(
                zip([p.id for p in to_read], pool.map(self.reader.read, to_read), strict=True)
            )
        passages = sum(self.store.add_paper(paper, texts.get(paper.id)) for paper in papers)
        return {"stats": {"full_texts": sum(1 for t in texts.values() if t), "passages": passages}}

    def dispatch(self, state: ResearchState) -> list[Send]:
        """Fan out: one researcher agent per sub-question, all running in parallel."""
        return [
            Send("researcher", {"question": state["question"], "sub_question": sq})
            for sq in state["plan"]
        ]

    # Researcher: answer one sub-question; search again if the evidence is thin ---------
    def research(self, task: ResearchTask) -> dict:
        question, sub = task["question"], task["sub_question"]
        note = self._take_note(question, sub, self._evidence(f"{sub.question} {sub.search_query}"))
        new_papers: list[Paper] = []
        follow_up = None
        for attempt in range(self.settings.max_follow_up_searches):
            if note.evidence_sufficient or not note.follow_up_query.strip():
                break
            follow_up = note.follow_up_query.strip()
            new_papers += self._search_more(follow_up, sub.question)
            last_try = attempt == self.settings.max_follow_up_searches - 1
            chunks = self._evidence(f"{sub.question} {follow_up}")
            note = self._take_note(question, sub, chunks, last_try=last_try)

        text = normalize_citations(note.finding)
        finding = Finding(
            sub_question=sub.question,
            text=text,
            paper_ids=cited_ids(text),
            follow_up_query=follow_up,
        )
        return {"findings": [finding], "papers": {paper.id: paper for paper in new_papers}}

    def _take_note(
        self, question: str, sub: SubQuestion, chunks: list[Chunk], last_try: bool = False
    ) -> ResearchNote:
        prompt = prompts.RESEARCHER_PROMPT.format(
            question=question,
            sub_question=sub.question,
            excerpts=format_excerpts(chunks, self.registry) or "(no excerpts found)",
        )
        if last_try:
            prompt += prompts.RESEARCHER_LAST_TRY
        return self.llm.structured(prompts.RESEARCHER_SYSTEM, prompt, ResearchNote)

    def _evidence(self, query: str) -> list[Chunk]:
        """Best passages for a query, at most two per paper so no single paper dominates."""
        limit = self.settings.chunks_per_question
        per_paper: dict[str, int] = {}
        selected = []
        for chunk in self.store.search(query, k=limit * 3):
            if per_paper.get(chunk.paper_id, 0) < 2:
                per_paper[chunk.paper_id] = per_paper.get(chunk.paper_id, 0) + 1
                selected.append(chunk)
            if len(selected) == limit:
                break
        return selected

    def _search_more(self, query: str, sub_question: str) -> list[Paper]:
        found = merge_papers(search_all(self.sources, [query], self.settings.results_per_query))
        fresh = [paper for paper in found if self.registry.find(paper) is None]
        added = []
        for paper in rank_papers(sub_question, fresh, self.store.embedder)[:FOLLOW_UP_PAPERS]:
            registered, is_new = self.registry.add(paper)
            if is_new:
                self.store.add_paper(registered)
                added.append(registered)
        return added

    # Writer: merge the notes into one report ------------------------------------------
    def write(self, state: ResearchState) -> dict:
        findings = ordered_findings(state)
        cited = sorted({pid for f in findings for pid in f.paper_ids}, key=id_number)
        papers = [self.registry.get(pid) for pid in cited if self.registry.get(pid)]
        prompt = prompts.WRITER_PROMPT.format(
            question=state["question"],
            catalog=format_catalog(papers) or "(none)",
            notes=format_notes(findings),
        )
        draft = normalize_citations(self.llm.text(prompts.WRITER_SYSTEM, prompt))
        return {"draft": draft, "revisions": 0}

    # Critic: fact-check every citation --------------------------------------------------
    def critique(self, state: ResearchState) -> dict:
        return {"review": fact_check(state["draft"], self.registry, self.store, self.llm)}

    def after_review(self, state: ResearchState) -> str:
        if state["review"].passed or state.get("revisions", 0) >= self.settings.max_revisions:
            return "finalize"
        return "revise"

    # Reviser: fix what the critic flagged -----------------------------------------------
    def revise(self, state: ResearchState) -> dict:
        review = state["review"]
        problems = [
            f'- "{claim.text}" (cites {", ".join(claim.citations)}): {claim.reason}'
            for claim in review.unsupported
        ]
        if review.invalid_citations:
            problems.append(
                "- These citation IDs match no paper and must be removed: "
                + ", ".join(review.invalid_citations)
            )
        prompt = prompts.REVISER_PROMPT.format(
            question=state["question"],
            notes=format_notes(ordered_findings(state)),
            problems="\n".join(problems),
            draft=state["draft"],
        )
        draft = normalize_citations(self.llm.text(prompts.REVISER_SYSTEM, prompt))
        return {"draft": draft, "revisions": state.get("revisions", 0) + 1}

    # Finalizer: numbered citations, fact-check notes, reference list ---------------------
    def finalize(self, state: ResearchState) -> dict:
        review = state["review"]
        known = self.registry.ids()
        draft = state["draft"]
        body = remove_citations(draft, [pid for pid in cited_ids(draft) if pid not in known])
        order = cited_ids(body)
        numbers = {pid: n for n, pid in enumerate(order, 1)}

        sections = [number_citations(body, order).strip()]
        if review.unsupported:
            notes = []
            for claim in review.unsupported:
                refs = ", ".join(str(numbers[pid]) for pid in claim.citations if pid in numbers)
                cite = f" [{refs}]" if refs else ""
                notes.append(f"- {claim.text}{cite}: {claim.reason}")
            sections.append(
                "## Fact-check notes\nThe critic agent could not confirm these statements "
                "from the cited sources:\n\n" + "\n".join(notes)
            )
        sections.append("## References\n" + reference_list([self.registry.get(p) for p in order]))
        sections.append(
            f"---\n*Generated by ScholarScout: {len(self.registry)} papers read, "
            f"{len(order)} cited, {review.supported_claims}/{len(review.claims)} cited "
            "statements confirmed by the critic agent.*"
        )
        return {"report": "\n\n".join(sections) + "\n", "cited": order}
