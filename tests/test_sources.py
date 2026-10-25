import httpx

from scholarscout.sources.arxiv import ArxivSource, build_query, parse_feed
from scholarscout.sources.openalex import OpenAlexSource, parse_works, rebuild_abstract

ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2312.00752v2</id>
    <title>Mamba: Linear-Time Sequence
      Modeling with Selective State Spaces</title>
    <summary>  Foundation models are almost universally based on
      the Transformer architecture.  </summary>
    <published>2023-12-01T18:01:34Z</published>
    <link href="https://arxiv.org/abs/2312.00752v2" rel="alternate" type="text/html"/>
    <link href="https://arxiv.org/pdf/2312.00752v2" rel="related" title="pdf"/>
    <arxiv:primary_category term="cs.LG"/>
    <author><name>Albert Gu</name></author>
    <author><name>Tri Dao</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/api/errors#incorrect_id_format</id>
    <title>Error</title>
  </entry>
</feed>"""

OPENALEX_PAGE = {
    "results": [
        {
            "id": "https://openalex.org/W1",
            "doi": "https://doi.org/10.48550/arxiv.2312.00752",
            "display_name": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
            "publication_year": 2023,
            "cited_by_count": 1500,
            "authorships": [{"author": {"display_name": "Albert Gu"}}],
            "abstract_inverted_index": {"Foundation": [0], "models": [1], "are": [2]},
            "primary_location": {
                "landing_page_url": "http://arxiv.org/abs/2312.00752",
                "pdf_url": "https://arxiv.org/pdf/2312.00752",
                "source": {"display_name": "arXiv (Cornell University)"},
            },
            "open_access": {"oa_url": "https://arxiv.org/pdf/2312.00752"},
        },
        {"id": "https://openalex.org/W2", "display_name": "No abstract", "doi": None},
    ]
}


def test_build_query_is_an_and_search_without_stopwords():
    assert build_query("How do state-space models compare to Transformers?") == (
        "all:state AND all:space AND all:models AND all:compare AND all:transformers"
    )


def test_parse_arxiv_feed():
    [paper] = parse_feed(ARXIV_FEED)
    assert paper.title == "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
    assert paper.abstract.startswith("Foundation models are almost")
    assert paper.arxiv_id == "2312.00752"
    assert paper.authors == ["Albert Gu", "Tri Dao"]
    assert paper.year == 2023
    assert paper.venue == "arXiv (cs.LG)"
    assert paper.pdf_url == "https://arxiv.org/pdf/2312.00752v2"


def test_arxiv_source_sends_and_query():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["query"] = request.url.params["search_query"]
        return httpx.Response(200, text=ARXIV_FEED)

    source = ArxivSource(http=httpx.Client(transport=httpx.MockTransport(handler)), min_interval=0)
    papers = source.search("mamba transformers")
    assert seen["query"] == "all:mamba AND all:transformers"
    assert len(papers) == 1


def test_rebuild_abstract_orders_words():
    assert rebuild_abstract({"world": [1], "hello": [0, 2]}) == "hello world hello"


def test_parse_openalex_works():
    [paper] = parse_works(OPENALEX_PAGE)  # the work without an abstract is skipped
    assert paper.arxiv_id == "2312.00752"
    assert paper.doi == "10.48550/arxiv.2312.00752"
    assert paper.citations == 1500
    assert paper.abstract == "Foundation models are"
    assert paper.venue == "arXiv (Cornell University)"


def test_openalex_source_passes_api_key():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.url.params)
        return httpx.Response(200, json=OPENALEX_PAGE)

    source = OpenAlexSource(
        http=httpx.Client(transport=httpx.MockTransport(handler)), api_key="secret"
    )
    assert len(source.search("How do state-space models scale?", max_results=5)) == 1
    assert seen["search"] == "How do state-space models scale"
    assert seen["api_key"] == "secret"
    assert seen["per-page"] == "5"
