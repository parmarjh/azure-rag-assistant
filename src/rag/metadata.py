from __future__ import annotations

import re
from datetime import datetime

from .models import Document


def _value(text: str, *labels: str) -> str | None:
    pat = rf"(?:{'|'.join(re.escape(x) for x in labels)})\s*:?\s*([^|\n]+)"
    m = re.search(pat, text, re.I)
    return m.group(1).strip() if m else None


def enrich_documents(documents: list[Document]) -> list[Document]:
    for d in documents:
        # Parsing puts the document header into the first section.
        header = d.sections[0].text[:1500] if d.sections else ""
        d.version = _value(header, "Version", "Template Version") or d.version
        d.effective_date = _value(header, "Effective", "Plan Year", "Last Updated") or d.effective_date
        d.supersedes = _value(header, "Supersedes") or d.supersedes
        title = d.title
        family = re.sub(r"\b(19|20)\d{2}\b", "", title, flags=re.I)
        family = re.sub(r"\b(rate card|version|v\d+(?:\.\d+)*)\b", "", family, flags=re.I)
        d.doc_family = re.sub(r"[^a-z0-9]+", "_", family.lower()).strip("_")
        d.doc_type = d.filename.rsplit(".", 1)[-1].lower()
        d.security_groups = [d.department.lower(), "all-staff"]
    groups: dict[str, list[Document]] = {}
    for d in documents:
        groups.setdefault(d.doc_family, []).append(d)
    for family_docs in groups.values():
        if len(family_docs) < 2:
            continue
        def date_key(doc: Document) -> datetime:
            m = re.search(r"(20\d{2})", f"{doc.effective_date} {doc.title}")
            return datetime(int(m.group(1)), 1, 1) if m else datetime.min
        family_docs.sort(key=date_key)
        current = family_docs[-1]
        for old in family_docs[:-1]:
            old.is_current = False
            old.superseded_by = current.doc_id
    return documents


def resolve_versions(documents: list[Document]) -> list[Document]:
    """Public alias for the data-driven document family resolver."""
    return enrich_documents(documents)
