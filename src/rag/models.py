from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Section:
    heading_path: str
    text: str
    page: int | None = None
    number: str | None = None


@dataclass
class Document:
    doc_id: str
    filename: str
    source_uri: str
    title: str
    department: str
    version: str = ""
    effective_date: str | None = None
    expiry_date: str | None = None
    supersedes: str | None = None
    doc_type: str = ""
    doc_family: str = ""
    is_current: bool = True
    superseded_by: str | None = None
    sections: list[Section] = field(default_factory=list)
    security_groups: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    filename: str
    source_uri: str
    department: str
    security_groups: list[str]
    doc_type: str
    doc_family: str
    version: str
    effective_date: str | None
    expiry_date: str | None
    is_current: bool
    supersedes: str | None
    superseded_by: str | None
    section_path: str
    section_number: str | None
    page: int | None
    chunk_index: int
    token_count: int
    keywords: list[str]
    header: str
    content: str
    embedding: list[float] = field(default_factory=list)


@dataclass
class Retrieved:
    chunk: Chunk
    score: float

    @property
    def chunk_id(self) -> str: return self.chunk.chunk_id
    @property
    def doc_id(self) -> str: return self.chunk.doc_id
    @property
    def filename(self) -> str: return self.chunk.filename
    @property
    def section_path(self) -> str: return self.chunk.section_path
    @property
    def content(self) -> str: return self.chunk.content


@dataclass
class Citation:
    chunk_id: str
    doc_id: str
    filename: str
    doc_title: str
    section_path: str
    page: int | None


@dataclass
class UserContext:
    user_id: str
    groups: list[str] = field(default_factory=lambda: ["all-staff"])
    department: str | None = None


@dataclass
class ChatTurn:
    question: str
    answer: str
    entities: dict[str, str] = field(default_factory=dict)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0


@dataclass
class IngestStats:
    documents: int
    chunks: int
    mode: str


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    retrieved: list[Retrieved]
    confidence: float
    abstained: bool
    clarification: str | None
    mode: str
    latency_ms: dict[str, float]
    usage: Usage
    provider: str
    rewritten_query: str
    filters: dict[str, Any]
