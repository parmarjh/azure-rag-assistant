from __future__ import annotations

import re

from .models import Chunk, Document, Section


def _tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def _keywords(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9$%.-]{2,}", text)
    return sorted(set(w.lower() for w in words if len(w) > 3))


def _make(doc: Document, text: str, section: Section, idx: int, with_header: bool) -> Chunk:
    header = (f"Document: {doc.title} | Dept: {doc.department} | Version {doc.version} | "
              f"Effective {doc.effective_date or 'unspecified'} | Section: {section.heading_path}"
              f"{f' (page {section.page})' if section.page else ''}")
    content = f"{header}\n{text}" if with_header else text
    return Chunk(f"{doc.doc_id}-{idx:04d}", doc.doc_id, doc.title, doc.filename, doc.source_uri,
                 doc.department, doc.security_groups, doc.doc_type, doc.doc_family, doc.version,
                 doc.effective_date, doc.expiry_date, doc.is_current, doc.supersedes,
                 doc.superseded_by, section.heading_path, section.number, section.page, idx,
                 len(_tokens(content)), _keywords(content), header, content)


def baseline_chunks(doc: Document, size: int = 1000) -> list[Chunk]:
    text = "\n".join(s.text for s in doc.sections)
    return [_make(doc, text[i:i + size], Section("Document", text[i:i + size]), n, False)
            for n, i in enumerate(range(0, len(text), size))]


def improved_chunks(
    doc: Document,
    target: int = 350,
    overlap: int = 80,
    with_header: bool = True,
) -> list[Chunk]:
    result: list[Chunk] = []
    n = 0
    for section in doc.sections:
        words = _tokens(section.text)
        if not words:
            continue
        if len(words) <= target:
            windows = [section.text]
        elif "|" in section.text:
            lines = [line.strip() for line in section.text.splitlines() if line.strip()]
            windows = []
            start = 0
            while start < len(lines):
                end = start
                count = 0
                while end < len(lines) and count + len(_tokens(lines[end])) <= target:
                    count += len(_tokens(lines[end]))
                    end += 1
                if end == start:
                    end += 1
                windows.append("\n".join(lines[start:end]))
                if end >= len(lines):
                    break
                overlap_count = 0
                next_start = end
                while next_start > start and overlap_count < overlap:
                    next_start -= 1
                    overlap_count += len(_tokens(lines[next_start]))
                if next_start == start:
                    next_start = end
                start = next_start
        else:
            step = max(1, target - overlap)
            windows = [words[i:i + target] for i in range(0, len(words), step)]
        for window in windows:
            text = window if isinstance(window, str) else " ".join(window)
            result.append(_make(doc, text, section, n, with_header))
            n += 1
    return result
