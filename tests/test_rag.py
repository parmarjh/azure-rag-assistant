from pathlib import Path

from rag.acl import allowed, build_filter
from rag.chunking import baseline_chunks, improved_chunks
from rag.config import get_config
from rag.guardrails import has_sufficient_evidence
from rag.models import UserContext
from rag.parsing import parse_directory
from rag.pipeline import RagPipeline
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
