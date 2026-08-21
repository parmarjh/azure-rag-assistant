from pathlib import Path

from rag.acl import allowed, build_filter
from rag.chunking import baseline_chunks, improved_chunks
from rag.config import get_config
from rag.guardrails import has_sufficient_evidence, validate_citations_and_numbers
from rag.models import ChatTurn, Citation, Document, Retrieved, Section, UserContext
from rag.parsing import parse_directory
from rag.pipeline import RagPipeline
from rag.query import understand
from rag.retrieval import reciprocal_rank_fusion

ROOT = Path(__file__).parents[1]
KB = ROOT / "data" / "KnowledgeBase"


def test_parse_and_chunk_modes():
    docs = parse_directory(KB)
    assert len(docs) == 11
    assert any("2.3" in s.heading_path for d in docs for s in d.sections)
    assert len(improved_chunks(docs[0])) >= 1
    assert len(baseline_chunks(docs[0])) >= 1


def test_version_resolution():
    docs = parse_directory(KB)
    pricing = [d for d in docs if "Pricing" in d.filename]
    assert any(d.is_current and "2026" in d.title for d in pricing)
    assert any(not d.is_current and "2025" in d.title for d in pricing)


def test_acl():
    p = RagPipeline.from_config(get_config())
    p.ingest(str(KB))
    sales = next(c for c in p.chunks if c.department == "Sales")
    assert allowed(sales, UserContext("u", ["sales"], "Sales"))
    assert not allowed(sales, UserContext("u", ["hr"], "HR"))
    assert build_filter(UserContext("u", ["hr"], "HR"))["groups"] == ["hr"]
    assert not any(
        c.department == "HR"
        for c in p.answer(
            "How many paid parental leave weeks are available?",
            user=UserContext("engineering", ["engineering", "all-staff"], "Engineering"),
        ).retrieved
    )
    denied = p.answer(
        "How many paid parental leave weeks are available?",
        user=UserContext("engineering", ["engineering", "all-staff"], "Engineering"),
    )
    assert denied.abstained


def test_rrf_and_guardrail():
    p = RagPipeline.from_config(get_config())
    p.ingest(str(KB))
    first = p.index.search("parental leave", p.embedder.embed(["parental leave"])[0],
                            {"is_current": True}, 3)
    fused = reciprocal_rank_fusion([first, list(reversed(first))])
    assert fused and {x.chunk_id for x in fused} == {x.chunk_id for x in first}
    assert has_sufficient_evidence("paid parental leave", first)
    assert not has_sufficient_evidence("stock ticker symbol", first)


def test_required_answers():
    p = RagPipeline.from_config(get_config())
    p.ingest(str(KB))
    leave = p.answer("How many days of paid parental leave do eligible employees get?")
    assert "12 weeks" in leave.text
    assert any(c.filename == "LeavePolicy.pdf" and "2.3" in c.section_path for c in leave.citations)
    price = p.answer("What is the price per seat for the Enterprise tier?")
    assert "$109" in price.text
    assert any(c.filename == "Pricing2026.pdf" for c in price.citations)
    assert p.answer("What is our company's stock ticker symbol?").abstained
    assert p.answer("What is the limit?").clarification


def test_improved_chunk_headers_tables_and_overlap():
    docs = parse_directory(KB)
    pricing = next(d for d in docs if d.filename == "Pricing2026.pdf")
    chunks = improved_chunks(pricing, target=20, overlap=5)
    assert chunks[0].header in chunks[0].content
    assert all(chunks[i].content.split()[-5:] for i in range(len(chunks)))
    discounts = next(d for d in docs if d.filename == "Discounts.xlsx")
    table_chunks = improved_chunks(discounts)
    assert any("5–24 seats" in chunk.content and "| 5 |" in chunk.content for chunk in table_chunks)
    section = Section("1 Topic", " ".join(f"word{i}" for i in range(180)))
    synthetic = Document("synthetic", "synthetic.txt", "", "Synthetic", "IT", sections=[section])
    overlapping = improved_chunks(synthetic, target=100, overlap=20)
    first_body = overlapping[0].content[len(overlapping[0].header):].split()
    second_body = overlapping[1].content[len(overlapping[1].header):].split()
    assert set(first_body[-20:]) & set(second_body[:20])


def test_follow_up_condensation():
    cases = [
        (
            [ChatTurn("What is the cancellation policy for the Enterprise plan?", "", {})],
            "What about the Starter tier?",
            ("starter", "cancellation"),
        ),
        (
            [ChatTurn("How many PTO days does a new employee accrue per year?", "", {})],
            "What about someone with 7 years of service?",
            ("PTO", "7 years"),
        ),
        (
            [ChatTurn("How many paid sick days do we get each year?", "", {})],
            "Do they carry over to next year?",
            ("sick", "carry over"),
        ),
        (
            [ChatTurn("What are the nightly hotel rate caps?", "", {})],
            "Which cities count as Tier 1?",
            ("hotel", "Tier 1"),
        ),
        (
            [
                ChatTurn("Is multi-factor authentication required for all accounts?", "", {}),
                ChatTurn("What about SMS?", "", {}),
            ],
            "Are there any exceptions?",
            ("multi-factor", "exceptions"),
        ),
    ]
    for history, question, expected in cases:
        rewritten = understand(question, history)["rewritten"].lower()
        assert all(term.lower() in rewritten for term in expected)


def test_ambiguity_facets_and_answer_metadata():
    pipeline = RagPipeline.from_config(get_config())
    pipeline.ingest(str(KB))
    clarification = pipeline.answer("What is the limit?")
    assert clarification.clarification
    assert "expense category limits" in clarification.text or "hotel nightly caps" in clarification.text
    answer = pipeline.answer("How many paid parental leave weeks are available?")
    assert answer.latency_ms["rewrite"] >= 0
    assert answer.latency_ms["search"] > 0
    assert answer.latency_ms["rerank"] > 0
    assert answer.latency_ms["generate"] > 0
    assert answer.latency_ms["total"] >= answer.latency_ms["search"]
    assert answer.usage.prompt_tokens > 0
    assert answer.usage.estimated_cost_usd > 0


def test_numeric_post_validation_rejects_unsupported_number():
    docs = parse_directory(KB)
    chunk = next(
        chunk for chunk in improved_chunks(next(d for d in docs if d.filename == "Pricing2026.pdf"))
        if "$109" in chunk.content
    )
    item = Retrieved(chunk, 1.0)
    citation = Citation(chunk.chunk_id, chunk.doc_id, chunk.filename, chunk.doc_title,
                         chunk.section_path, chunk.page)
    assert validate_citations_and_numbers("Enterprise costs $109 [1]", [citation], [item])
    assert not validate_citations_and_numbers("Enterprise costs $999 [1]", [citation], [item])
    assert not validate_citations_and_numbers("Enterprise costs $109 [2]", [citation], [item])
