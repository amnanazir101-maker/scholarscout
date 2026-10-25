"""ScholarScout web app. Run with: streamlit run app.py"""

from dataclasses import replace

import anthropic
import streamlit as st
from dotenv import load_dotenv

from scholarscout.agents import ResearchError
from scholarscout.config import Settings
from scholarscout.graph import ResearchAssistant, describe
from scholarscout.llm import DEFAULT_MODELS, LLMError, make_llm
from scholarscout.store import FastEmbedEmbedder

load_dotenv()
st.set_page_config(page_title="ScholarScout", page_icon="🔭", layout="wide")

EXAMPLES = [
    "How do state space models such as Mamba compare with transformers for long sequences?",
    "What are the main techniques for reducing hallucinations in large language models?",
    "How are retrieval-augmented generation systems evaluated?",
]
PROVIDERS = {"gemini": "Gemini (free tier)", "claude": "Claude (paid API)"}
KEY_NAMES = {"gemini": "GEMINI_API_KEY", "claude": "ANTHROPIC_API_KEY"}
EFFORTS = {"gemini": ["low", "medium", "high"], "claude": ["low", "medium", "high", "xhigh", "max"]}


@st.cache_resource(show_spinner="Loading the embedding model (first run downloads ~90 MB)...")
def load_embedder(model_name: str) -> FastEmbedEmbedder:
    return FastEmbedEmbedder(model_name)


defaults = Settings.from_env()

with st.sidebar:
    st.header("Settings")
    provider = st.radio(
        "LLM provider",
        list(PROVIDERS),
        index=list(PROVIDERS).index(defaults.provider) if defaults.provider in PROVIDERS else 0,
        format_func=PROVIDERS.get,
    )
    api_key = st.text_input(
        f"{PROVIDERS[provider].split()[0]} API key",
        type="password",
        help=f"Used only for this session. Leave empty if {KEY_NAMES[provider]} is set "
        "on the server. Free Gemini keys: https://aistudio.google.com/apikey",
    )
    model = st.text_input("Model", value=defaults.model or DEFAULT_MODELS[provider])
    efforts = EFFORTS[provider]
    effort = st.select_slider(
        "Thinking effort",
        efforts,
        value=defaults.effort if defaults.effort in efforts else "medium",
    )
    max_papers = st.slider("Papers to keep", 5, 30, defaults.max_papers)
    full_texts = st.slider("Full-text PDFs to read", 0, 8, defaults.full_text_papers)
    revisions = st.slider("Critic revision rounds", 0, 3, defaults.max_revisions)
    st.caption("Sources: arXiv and OpenAlex. Embeddings run locally on the CPU.")

st.title("🔭 ScholarScout")
st.write(
    "A team of AI agents searches arXiv and OpenAlex, reads the papers, writes a literature "
    "review and fact-checks every citation before you see it."
)

st.session_state.setdefault("question", "")
for column, example in zip(st.columns(len(EXAMPLES)), EXAMPLES, strict=True):
    if column.button(example, width="stretch"):
        st.session_state.question = example
question = st.text_area("Research question", key="question", height=90)
start = st.button("Start research", type="primary", disabled=not question.strip())

if start:
    settings = replace(
        defaults,
        provider=provider,
        model=model.strip(),
        effort=effort,
        max_papers=max_papers,
        full_text_papers=full_texts,
        max_revisions=revisions,
    )
    st.session_state.pop("result", None)
    try:
        llm = make_llm(settings, api_key=api_key.strip() or None)
        embedder = load_embedder(settings.embedding_model)
        assistant = ResearchAssistant(settings, llm=llm, embedder=embedder)
        with st.status("Agents at work... (usually 2-5 minutes)", expanded=True) as status:
            for agent, update in assistant.stream(question.strip()):
                if agent == "done":
                    st.session_state.result = update
                else:
                    status.write(describe(agent, update))
            status.update(label="Research finished", state="complete", expanded=False)
    except anthropic.AuthenticationError:
        st.error("Claude rejected the API key.")
    except (LLMError, ResearchError, anthropic.APIError) as exc:
        st.error(str(exc))

result = st.session_state.get("result")
if result:
    stats = result.stats
    columns = st.columns(5)
    columns[0].metric("Papers screened", stats.get("candidates", 0))
    columns[1].metric("Papers read", stats["papers_read"])
    columns[2].metric("Papers cited", stats["papers_cited"])
    columns[3].metric(
        "Statements verified", f"{stats['claims_supported']}/{stats['claims_checked']}"
    )
    columns[4].metric("Time", f"{stats['seconds']:.0f} s")
    if stats.get("llm_calls"):
        cost = stats.get("cost_usd")
        price = f", about ${cost:.2f}" if cost is not None else ""
        if cost is None and stats.get("provider") == "gemini":
            price = ", free tier"
        models = ", ".join(stats.get("models_used") or [stats["model"]])
        st.caption(
            f"{models}: {stats['llm_calls']} calls, "
            f"{stats['input_tokens'] + stats['output_tokens']:,} tokens{price}"
        )

    numbers = {paper.id: n for n, paper in enumerate(result.cited, 1)}
    report_tab, sources_tab, check_tab, notes_tab = st.tabs(
        ["Report", "Sources", "Fact-check", "Research notes"]
    )
    with report_tab:
        st.markdown(result.report)
        st.download_button(
            "Download Markdown", result.report, "literature-review.md", "text/markdown"
        )
    with sources_tab:
        st.dataframe(
            [
                {
                    "#": n,
                    "Title": paper.title,
                    "Year": paper.year,
                    "Venue": paper.venue,
                    "Citations": paper.citations,
                    "Link": paper.url,
                }
                for n, paper in enumerate(result.cited, 1)
            ],
            column_config={"Link": st.column_config.LinkColumn()},
            hide_index=True,
            width="stretch",
        )
        with st.expander(f"All {len(result.papers)} papers the agents read"):
            st.dataframe(
                [
                    {"ID": p.id, "Title": p.title, "Year": p.year, "Score": p.score, "Link": p.url}
                    for p in result.papers
                ],
                column_config={"Link": st.column_config.LinkColumn()},
                hide_index=True,
                width="stretch",
            )
    with check_tab:
        review = result.review
        st.write(
            f"The critic agent checked {len(review.claims)} cited statements against the "
            f"abstracts and full text of the cited papers: {review.support_rate:.0%} supported."
        )
        for claim in review.claims:
            marker = "✅" if claim.supported else "⚠️"
            refs = ", ".join(str(numbers[pid]) for pid in claim.citations if pid in numbers)
            st.markdown(f"{marker} {claim.text}" + (f" [{refs}]" if refs else ""))
            st.caption(claim.reason)
    with notes_tab:
        st.caption("What each researcher agent found. IDs refer to 'All papers' under Sources.")
        for finding in result.findings:
            st.subheader(finding.sub_question)
            st.markdown(finding.text)
            if finding.follow_up_query:
                st.caption(f'Ran a follow-up search: "{finding.follow_up_query}"')
