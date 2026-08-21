from __future__ import annotations

import json
import logging
import os
import time

log = logging.getLogger("rag")
_azure_configured = False


def _configure_azure_monitor() -> None:
    global _azure_configured
    if _azure_configured or not os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
        return
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
    except ImportError:
        log.warning("azure-monitor-opentelemetry is unavailable; using structured logs")
    else:
        configure_azure_monitor(
            connection_string=os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"]
        )
    _azure_configured = True


class Span:
    def __init__(self, name: str):
        self.name = name
        self.start = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000
        log.info("rag_stage stage=%s latency_ms=%.2f", self.name, self.elapsed_ms)


def emit_request(record: dict) -> None:
    _configure_azure_monitor()
    log.info("rag_request %s", json.dumps(record, sort_keys=True, default=str))
