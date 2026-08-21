from __future__ import annotations

import argparse
import json
import os

from .config import get_config
from .pipeline import RagPipeline


def _pipeline(mode: str) -> RagPipeline:
    return RagPipeline.from_config(get_config(mode))


def main():
    parser = argparse.ArgumentParser(prog="rag")
    sub = parser.add_subparsers(dest="command", required=True)
    ing = sub.add_parser("ingest")
    ing.add_argument("--source", required=True)
    ing.add_argument("--mode", choices=["baseline", "improved"], default="improved")
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--mode", choices=["baseline", "improved"], default="improved")
    ask.add_argument("--session-id")
    chat = sub.add_parser("chat")
    chat.add_argument("--mode", choices=["baseline", "improved"], default="improved")
    sub.add_parser("reindex")
    args = parser.parse_args()
    if args.command == "ingest":
        stats = _pipeline(args.mode).ingest(args.source)
        print(json.dumps(stats.__dict__, indent=2))
    elif args.command == "ask":
        p = _pipeline(args.mode)
        p.ingest(os.getenv("RAG_SOURCE", "data/KnowledgeBase"))
        result = p.answer(args.question, args.session_id)
        print(f"Answer: {result.text}")
        if result.citations:
            print("Citations:")
            for i, citation in enumerate(result.citations, 1):
                print(f"  [{i}] {citation.filename} — {citation.section_path}"
                      f"{f' (page {citation.page})' if citation.page else ''}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Abstained: {result.abstained}")
        if result.clarification:
            print(f"Clarification: {result.clarification}")
        print(f"Mode: {result.mode}")
        print(f"Provider: {result.provider}")
    elif args.command == "chat":
        print("Interactive chat. Press Ctrl-D to exit.")
        p = _pipeline(args.mode)
        p.ingest(os.getenv("RAG_SOURCE", "data/KnowledgeBase"))
        while True:
            try:
                question = input("> ")
            except EOFError:
                break
            print(p.answer(question).text)
    else:
        print("Use ingest or ask; reindex is reserved for managed deployments.")


if __name__ == "__main__":
    main()
