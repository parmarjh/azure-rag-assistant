from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from rag.config import get_config
from rag.models import UserContext
from rag.pipeline import RagPipeline

app = FastAPI(title="Northwind Enterprise Knowledge Assistant", version="0.1.0")
_pipelines: dict[str, RagPipeline] = {}


def pipeline(mode: str) -> RagPipeline:
    if mode not in _pipelines:
        p = RagPipeline.from_config(get_config(mode))
        source = os.getenv("RAG_SOURCE", "data/KnowledgeBase")
        if Path(source).exists():
            p.ingest(source)
        _pipelines[mode] = p
    return _pipelines[mode]


@app.get("/health")
def health():
    return {"status": "ok", "provider": get_config().provider}


@app.get("/config")
def config():
    c = get_config()
    return {"mode": c.mode, "provider": c.provider, "index": c.search_index}


@app.post("/chat")
def chat(payload: dict):
    question = payload.get("question") or payload.get("message")
    if not question:
        raise HTTPException(400, "question is required")
    user_data = payload.get("user")
    user = UserContext(**user_data) if user_data else None
    result = pipeline(payload.get("mode", "improved")).answer(
        question, payload.get("session_id"), user)
    return {
        "text": result.text,
        "citations": [c.__dict__ for c in result.citations],
        "retrieved": [{"chunk_id": r.chunk_id, "doc_id": r.doc_id, "filename": r.filename,
                       "section_path": r.section_path, "score": r.score, "content": r.content}
                      for r in result.retrieved],
        "confidence": result.confidence, "abstained": result.abstained,
        "clarification": result.clarification, "mode": result.mode,
        "latency_ms": result.latency_ms, "usage": result.usage.__dict__,
        "provider": result.provider, "rewritten_query": result.rewritten_query,
        "filters": result.filters,
    }


@app.post("/ingest")
def ingest(payload: dict):
    source = payload.get("source", "data/KnowledgeBase")
    mode = payload.get("mode", "improved")
    result = RagPipeline.from_config(get_config(mode)).ingest(source)
    return result.__dict__


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
