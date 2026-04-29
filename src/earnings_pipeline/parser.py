import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SECTION_HEADERS = {
    "Presentation Operator Message",
    "Presenter Speech",
    "Question and Answer Operator Message",
    "Question",
    "Answer",
}

HEADER_RE = re.compile(
    r"^(?P<company>.+?),\s*Q(?P<q>\d)\s*(?P<y>\d{4}).*?Earnings Call.*?(?P<date>[A-Z][a-z]+ \d{1,2},\s*\d{4})"
)
ROLE_LINE_STRICT_RE = re.compile(r"^(Executives|Analysts|Operator)\s*-\s*.+$")


@dataclass
class Transcript:
    ticker: str
    quarter: str
    call_date: Optional[str]
    company: str
    prepared: List[Dict]
    qa: List[Dict]
    raw_path: str


def _filename_meta(path: Path) -> Tuple[str, str]:
    stem = path.stem
    ticker, _, q = stem.partition("_")
    return ticker, q


def _blocks(text: str):
    lines = text.splitlines()
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        if line in SECTION_HEADERS:
            section = line
            i += 1
            role = lines[i].strip() if i < n else ""
            i += 1
            buf = []
            while i < n and lines[i].strip() not in SECTION_HEADERS:
                buf.append(lines[i])
                i += 1
            yield section, role, "\n".join(buf).strip()
        else:
            i += 1


def parse_transcript(path: Path) -> Transcript:
    text = path.read_text(errors="ignore")
    first = text.splitlines()[0] if text.splitlines() else ""
    m = HEADER_RE.match(first)
    if m:
        company = m.group("company").strip()
        date = datetime.strptime(m.group("date").replace("  ", " "), "%b %d, %Y").strftime("%Y-%m-%d")
    else:
        company, date = "", None

    prepared, qa = [], []
    current_q = None
    in_qa = False

    for section, role, body in _blocks(text):
        if section == "Question and Answer Operator Message":
            in_qa = True
            continue
        if not in_qa and section == "Presenter Speech":
            prepared.append({"role": role, "text": body})
        elif in_qa and section == "Question":
            current_q = {"q_role": role, "question": body, "a_role": None, "answer": None}
            qa.append(current_q)
        elif in_qa and section == "Answer":
            if current_q is None or current_q["answer"] is not None:
                current_q = {"q_role": None, "question": "", "a_role": role, "answer": body}
                qa.append(current_q)
            else:
                current_q["a_role"] = role
                current_q["answer"] = body

    ticker, quarter = _filename_meta(path)
    return Transcript(ticker=ticker, quarter=quarter, call_date=date, company=company, prepared=prepared, qa=qa, raw_path=str(path))


def split_by_role_lines(body: str):
    lines = (body or "").splitlines()
    turns = []
    current_role = None
    buf = []

    def _flush():
        if current_role and any(x.strip() for x in buf):
            turns.append({"role": current_role.strip(), "text": "\n".join(buf).strip()})

    for ln in lines:
        s = ln.strip()
        if ROLE_LINE_STRICT_RE.match(s):
            _flush()
            current_role = s
            buf = []
        else:
            buf.append(ln)

    _flush()
    if not turns and body.strip():
        turns = [{"role": "Unknown", "text": body.strip()}]
    return turns


def parse_transcript_v2(path: Path) -> Transcript:
    text = path.read_text(errors="ignore")
    first = text.splitlines()[0] if text.splitlines() else ""
    m = HEADER_RE.match(first)
    if m:
        company = m.group("company").strip()
        date = datetime.strptime(m.group("date").replace("  ", " "), "%b %d, %Y").strftime("%Y-%m-%d")
    else:
        company, date = "", None

    prepared, qa = [], []
    in_qa = False
    current_q = None

    for section, role, body in _blocks(text):
        if section == "Question and Answer Operator Message":
            in_qa = True
            continue

        if not in_qa and section == "Presenter Speech":
            turns = split_by_role_lines("\n".join([role, body]).strip())
            for tr in turns:
                prepared.append({"role": tr["role"], "text": tr["text"]})
        elif in_qa and section == "Question":
            q_turns = split_by_role_lines("\n".join([role, body]).strip())
            q_role = q_turns[0]["role"] if q_turns else role
            q_text = "\n".join(t["text"] for t in q_turns if t["text"].strip()) if q_turns else body
            current_q = {"q_role": q_role, "question": q_text, "a_role": None, "answer": None}
            qa.append(current_q)
        elif in_qa and section == "Answer":
            a_turns = split_by_role_lines("\n".join([role, body]).strip())
            a_role = a_turns[0]["role"] if a_turns else role
            a_text = "\n".join(t["text"] for t in a_turns if t["text"].strip()) if a_turns else body
            if current_q is None or current_q["answer"] is not None:
                current_q = {"q_role": None, "question": "", "a_role": a_role, "answer": a_text}
                qa.append(current_q)
            else:
                current_q["a_role"] = a_role
                current_q["answer"] = a_text

    ticker, quarter = _filename_meta(path)
    return Transcript(ticker=ticker, quarter=quarter, call_date=date, company=company, prepared=prepared, qa=qa, raw_path=str(path))
