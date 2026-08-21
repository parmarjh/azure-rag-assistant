from __future__ import annotations

import re
from pathlib import Path

from .models import Document, Section

try:
    import pdfplumber
except ImportError:  # pragma: no cover
    pdfplumber = None
try:
    from docx import Document as DocxDocument
except ImportError:  # pragma: no cover
    DocxDocument = None
try:
    import openpyxl
except ImportError:  # pragma: no cover
    openpyxl = None

HEADING = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+(.+?)\s*$")
BOILERPLATE = re.compile(
    r"(?i)^\s*(?:[-—]\s*)?(?:internal use only|page\s+\d+|"
    r"\(cid:\d+\))\s*$"
)


def _clean_text(text: str) -> str:
    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"\(cid:\d+\)", "", raw_line, flags=re.I)
        line = re.sub(r"(?i)\s*[-—]\s*internal use only\b", "", line)
        line = re.sub(r"(?i)\bpage\s+\d+\b", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if not line or BOILERPLATE.match(line) or line.casefold() == "northwind traders, inc.":
            continue
        lines.append(line)
    return "\n".join(lines)


def _title(text: str, path: Path) -> str:
    first = next((x.strip() for x in text.splitlines() if x.strip()), path.stem)
    return re.sub(r"\s*\|\s*Northwind.*$", "", first).strip()


def _sections(text: str, pages: list[int] | None = None) -> list[Section]:
    lines = [re.sub(r"\s+", " ", x).strip() for x in text.splitlines()]
    sections: list[Section] = []
    current: Section | None = None
    for i, line in enumerate(lines):
        if not line:
            continue
        m = HEADING.match(line)
        if m and (len(line) < 140 or m.group(1).count(".") > 0):
            if current and current.text.strip():
                sections.append(current)
            current = Section(m.group(1) + " " + m.group(2), "", pages[i] if pages and i < len(pages) else None, m.group(1))
        elif current is None:
            current = Section("Document", line)
        else:
            current.text += ("\n" if current.text else "") + line
    if current and current.text.strip():
        sections.append(current)
    return sections or [Section("Document", text)]


def parse_file(path: str | Path, department: str | None = None) -> Document:
    p = Path(path)
    department = department or p.parent.name
    text = ""
    pages: list[int] = []
    if p.suffix.lower() == ".pdf" and pdfplumber:
        with pdfplumber.open(p) as pdf:
            for page_no, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text() or ""
                text += page_text + "\n"
                pages.extend([page_no] * max(1, len(page_text.splitlines())))
    elif p.suffix.lower() == ".docx" and DocxDocument:
        d = DocxDocument(p)
        text = "\n".join(x.text for x in d.paragraphs if x.text.strip())
        for table in d.tables:
            text += "\n" + "\n".join(" | ".join(c.text.strip() for c in row.cells) for row in table.rows)
    elif p.suffix.lower() == ".xlsx" and openpyxl:
        book = openpyxl.load_workbook(p, data_only=True)
        blocks = []
        for sheet in book.worksheets:
            rows = []
            for row in sheet.iter_rows(values_only=True):
                vals = [str(x) if x is not None else "" for x in row]
                if any(vals):
                    rows.append("| " + " | ".join(vals) + " |")
            blocks.append(f"Sheet: {sheet.title}\n" + "\n".join(rows))
        text = "\n\n".join(blocks)
    else:
        text = p.read_text(errors="replace")
    text = _clean_text(text)
    title = _title(text, p)
    doc_id = re.sub(r"[^a-z0-9]+", "-", p.stem.lower()).strip("-")
    return Document(doc_id, p.name, str(p), title, department, sections=_sections(text, pages))


def parse_directory(source_dir: str | Path) -> list[Document]:
    root = Path(source_dir)
    docs = [parse_file(p) for p in sorted(root.rglob("*")) if p.suffix.lower() in {".pdf", ".docx", ".xlsx"}]
    from .metadata import enrich_documents
    return enrich_documents(docs)
