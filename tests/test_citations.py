from scholarscout.citations import (
    cited_ids,
    extract_claims,
    normalize_citations,
    number_citations,
    reference_list,
    remove_citations,
)
from tests.fakes import PAPERS, WRITER_DRAFT


def test_grouped_markers_are_split():
    assert normalize_citations("Fast [P1, P3] and small [P2; P4].") == (
        "Fast [P1][P3] and small [P2][P4]."
    )


def test_cited_ids_keep_first_appearance_order():
    assert cited_ids("A [P3]. B [P1][P3]. C [P2, P1].") == ["P3", "P1", "P2"]


def test_remove_citations_tidies_punctuation():
    assert remove_citations("Claim [P1] [P99].", ["P99"]) == "Claim [P1]."


def test_claims_from_prose_bullets_and_tables():
    claims, cited, total = extract_claims(WRITER_DRAFT)
    texts = [claim.text for claim in claims]

    assert "Selective state space models scale linearly with sequence length." in texts
    table_row = next(claim for claim in claims if claim.text.startswith("Approach: Mamba"))
    assert table_row.citations == ["P1", "P2"]
    assert "Papers" not in table_row.text
    # 2 TL;DR + 3 key-finding sentences + 1 table row; the open-problem bullet is not counted
    assert (cited, total) == (6, 6)


def test_citation_only_fragment_joins_previous_sentence():
    claims, cited, total = extract_claims("Mamba is fast on GPUs. [P1]\n\nNo citation here at all.")
    assert [claim.text for claim in claims] == ["Mamba is fast on GPUs."]
    assert claims[0].citations == ["P1"]
    assert (cited, total) == (1, 2)


def test_abbreviations_do_not_split_sentences():
    claims, _, _ = extract_claims("Some models, e.g. Mamba, scale linearly [P1].")
    assert len(claims) == 1


def test_number_citations_merges_runs_in_reference_order():
    text = "A [P7]. B [P2][P7]. C [P2] [P9]."
    assert number_citations(text, ["P7", "P2"]) == "A [1]. B [1, 2]. C [2]."


def test_reference_list_is_numbered():
    papers = [PAPERS[0].model_copy(update={"url": "https://arxiv.org/abs/2312.00752"})]
    assert reference_list(papers) == (
        "1. Ada Lovelace and Alan Turing (2023). Mamba: Linear-Time Sequence Modeling with "
        "Selective State Spaces. https://arxiv.org/abs/2312.00752"
    )
