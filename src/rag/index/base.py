from __future__ import annotations

from typing import Protocol

from ..models import Chunk, Retrieved


class SearchIndex(Protocol):
    def create(self): ...
    def upload(self, chunks: list[Chunk]): ...
    def search(self, query: str, vector: list[float] | None, filters: dict,
               top_k: int) -> list[Retrieved]: ...
