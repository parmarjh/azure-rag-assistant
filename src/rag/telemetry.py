from __future__ import annotations

import logging
import time

log = logging.getLogger("rag")


class Span:
    def __init__(self, name: str):
        self.name, self.start = name, time.perf_counter()
    def __enter__(self): return self
    def __exit__(self, *_):
        log.info("rag_stage stage=%s latency_ms=%.2f", self.name,
                 (time.perf_counter() - self.start) * 1000)
