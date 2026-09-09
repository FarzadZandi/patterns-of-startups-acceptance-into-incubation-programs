"""Transcript segmentation for the Campus Founders NLP + fs-QCA pipeline."""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, NamedTuple

import pandas as pd

LOGGER = logging.getLogger(__name__)

QUESTION_WORDS = (
    "What",
    "How",
    "Why",
    "When",
    "Where",
    "Who",
    "Can",
    "Could",
    "Would",
    "Is",
    "Are",
    "Do",
    "Does",
    "Did",
    "Have",
    "Has",
    "Will",
    "Shall",
)

EVALUATOR_LABEL_RE = re.compile(
    r"^\s*(?:(?:Mentor|Evaluator|Panel|Judge|Audience|Interviewer)\s*:|(?=.{1,120}:)(?=.*(?:CF|Campus Founders|AI Founders|ZFHN|Pyzalski|Andrea Muth|Tristan|Nico|Patrick|Samer|Etienne|Sebastian)).{1,120}:|.{1,120}(?:@\d+:\d+|\(?\d+:\d+\)?)\s*:)",
    re.IGNORECASE,
)
SPEAKER_LINE_RE = re.compile(
    r"^\s*(?P<speaker>.+?)\s*(?:\[[0-9:.]+\]|\d{1,2}:\d{2}(?::\d{2})?)\s*:?\s*(?P<text>.*)$"
)
QA_HEADING_RE = re.compile(r"^\s*(q\s*&\s*a|q&a|questions?)\s*[:.-]?\s*$", re.IGNORECASE)
QA_START_RE = re.compile(
    r"\b(q\s*&\s*a|q&a|questions?|your questions|answer any questions)\b",
    re.IGNORECASE,
)
WHITESPACE_RE = re.compile(r"\s+")
HTML_TAG_RE = re.compile(r"<[A-Za-z][^>]*>")
_SENT_TOKENIZER = "uninitialized"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self._parts.append(data.strip())

    def text(self) -> str:
        return "\n".join(self._parts)


class TranscriptFile(NamedTuple):
    path: Path
    batch: int
    startup_id: str
    role_policy: str
    source: str
    key: str


def read_transcript(path: Path, parse_html: bool = False) -> str:
    """Read a transcript, optionally extracting readable text from HTML."""
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    if parse_html or HTML_TAG_RE.search(raw):
        parser = _TextExtractor()
        parser.feed(raw)
        extracted = parser.text()
        if extracted:
            return extracted
    return raw


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", " ")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"//", " ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def strip_known_suffixes(stem: str, batch: int) -> str:
    """Extract startup_id from batch-specific filename conventions."""
    startup_id = stem.strip()

    if batch == 3:
        suffixes = [
            r"\s+Feedback_transcript_UPDATED$",
            r"\s+Feedback_transcription_UPDATED$",
            r"\s+Feedback\s+File\s+missing$",
            r"\s+Pitch\s+\+\s+Q&A_transcript_UPDATED$",
            r"\s+Pitch\s+\+\s+Q&A_transcription_UPDATED$",
            r"\s+Pitch\s+\+\s+Q&A_transcription$",
            r"\s+Pitch\s+html_updated$",
            r"\s+Pitch_transcript_UPDATED$",
            r"\s+Pitch_transciprtion_UPDATED$",
            r"\s+Q&A_transcript_UPDATED$",
            r"_transcript_UPDATED$",
            r"_transcription_UPDATED$",
        ]
    elif batch == 4:
        suffixes = [
            r"Feedback\d*_transcript_UPDATED$",
            r"Pitch_transcript_UPDATED$",
            r"_transcript_UPDATED$",
        ]
    else:
        suffixes = []

    for suffix in suffixes:
        startup_id = re.sub(suffix, "", startup_id, flags=re.IGNORECASE).strip()

    return normalize_text(startup_id)


def startup_key(startup_id: str) -> str:
    key = re.sub(r"\b(gmbh|ug|ltd|limited|inc)\b\.?", "", startup_id, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "", key.casefold())


def classify_file(path: Path, batch: int) -> tuple[str, str, str]:
    """Return startup_id, role policy, and source for a raw transcript file."""
    stem = path.stem
    lower_name = path.name.lower()
    startup_id = strip_known_suffixes(stem, batch)

    if batch == 3:
        if "feedback" in lower_name and "missing" in lower_name:
            return startup_id, "missing_feedback", "missing_feedback"
        if "feedback" in lower_name:
            return startup_id, "evaluator", "feedback"
        if "pitch + q&a" in lower_name or lower_name.endswith("_transcript_updated.txt"):
            return startup_id, "founder_qa", "pitch"
        if "html_updated" in lower_name or "pitch" in lower_name or "q&a" in lower_name:
            return startup_id, "founder_qa", "pitch"
        return startup_id, "founder_qa", "pitch"

    if batch == 4:
        if "feedback" in lower_name:
            return startup_id, "evaluator", "feedback"
        if "pitch" in lower_name:
            return startup_id, "founder", "pitch"
        return startup_id, "combined", "combined"

    return startup_id, "combined", "combined"


def batch_from_path(path: Path) -> int | None:
    match = re.search(r"Batch\s*0?([345])", str(path), re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def plain_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        cleaned = normalize_text(line)
        if cleaned:
            if re.match(r"^Batch\s+#?\d+\b", cleaned, re.IGNORECASE):
                continue
            if re.match(r"^\(\d{1,2}\s+[A-Za-z]{3}\s+\d{4}:", cleaned):
                continue
            lines.append(cleaned)
    return lines


def is_evaluator_turn(line: str) -> bool:
    if EVALUATOR_LABEL_RE.match(line):
        return True

    cleaned = re.sub(r"^[^:]{1,80}:\s*", "", line).strip().lstrip("'\" ")
    return cleaned.startswith(QUESTION_WORDS) and cleaned.endswith("?")


def detect_qa_start(lines: list[str]) -> int | None:
    for idx, line in enumerate(lines):
        if QA_HEADING_RE.match(line):
            return idx + 1
        if QA_START_RE.search(line) and "thank" in line.lower():
            return idx + 1
        if re.search(r"\b(your questions|open for questions|do you have any questions)\b", line, re.IGNORECASE):
            return idx + 1
        if re.search(r"\bquestions?\s*\?\s*$", line, re.IGNORECASE):
            return idx + 1
    return None


def strip_speaker_prefix(line: str) -> str:
    match = SPEAKER_LINE_RE.match(line)
    if match:
        return normalize_text(match.group("text"))
    return normalize_text(line)


def token_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def get_sentence_tokenizer():
    global _SENT_TOKENIZER
    if _SENT_TOKENIZER != "uninitialized":
        return _SENT_TOKENIZER

    try:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt")
            _SENT_TOKENIZER = nltk.sent_tokenize
            return _SENT_TOKENIZER
        except LookupError:
            _SENT_TOKENIZER = False
            return None
    except Exception:
        _SENT_TOKENIZER = False
        return None


def split_sentences(text: str) -> list[str]:
    tokenizer = get_sentence_tokenizer()
    if tokenizer:
        try:
            return [normalize_text(sentence) for sentence in tokenizer(text)]
        except LookupError:
            pass
    return [normalize_text(sentence) for sentence in re.split(r"(?<=[.!?])\s+", text)]


def make_segments(
    startup_id: str,
    batch: int,
    role_policy: str,
    source: str,
    text: str,
) -> list[dict[str, object]]:
    lines = plain_lines(text)
    qa_start = detect_qa_start(lines) if role_policy in {"founder_qa", "combined"} else None
    rows: list[dict[str, object]] = []

    for idx, line in enumerate(lines):
        current_source = source
        role = "founder"

        if role_policy == "evaluator":
            role = "evaluator"
        elif role_policy == "combined":
            current_source = "combined"
            if qa_start is not None and idx >= qa_start and is_evaluator_turn(line):
                role = "evaluator"
        elif role_policy == "founder_qa":
            if qa_start is not None and idx >= qa_start:
                current_source = "qa"
                role = "evaluator" if is_evaluator_turn(line) else "founder"
            else:
                current_source = "pitch"
        elif role_policy == "founder":
            role = "founder"
            current_source = "pitch"

        cleaned_line = strip_speaker_prefix(line)
        if not cleaned_line:
            continue

        for sentence in split_sentences(cleaned_line):
            if token_count(sentence) < 5:
                continue
            rows.append(
                {
                    "startup_id": str(startup_id),
                    "batch": int(batch),
                    "role": role,
                    "source": current_source,
                    "text": sentence,
                }
            )

    return rows


def iter_transcript_files(raw_dir: Path) -> Iterable[Path]:
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and not path.name.startswith("~$"):
            yield path


def collect_transcript_files(raw_dir: Path) -> list[TranscriptFile]:
    records: list[TranscriptFile] = []
    for path in iter_transcript_files(raw_dir):
        batch = batch_from_path(path)
        if batch is None:
            LOGGER.warning("Skipping file outside Batch 03/04/05 folders: %s", path)
            continue

        startup_id, role_policy, source = classify_file(path, batch)
        if role_policy == "missing_feedback":
            continue
        records.append(
            TranscriptFile(
                path=path,
                batch=batch,
                startup_id=startup_id,
                role_policy=role_policy,
                source=source,
                key=f"{batch}:{startup_key(startup_id)}",
            )
        )
    return records


def canonical_startup_ids(records: list[TranscriptFile]) -> dict[str, str]:
    choices: dict[str, list[TranscriptFile]] = defaultdict(list)
    for record in records:
        choices[record.key].append(record)

    canonical: dict[str, str] = {}
    for key, grouped in choices.items():
        preferred = sorted(
            grouped,
            key=lambda item: (
                item.source == "feedback",
                item.source == "missing_feedback",
                len(item.startup_id),
                item.startup_id.casefold(),
            ),
        )[0]
        canonical[key] = preferred.startup_id
    return canonical


def warn_missing_feedback(files_by_batch_startup: dict[tuple[int, str], set[str]]) -> None:
    for (batch, startup_id), sources in sorted(files_by_batch_startup.items()):
        if batch == 3 and ("pitch" in sources or "qa" in sources) and "feedback" not in sources:
            LOGGER.warning("Batch 03 startup '%s' has no Feedback file; skipping evaluator side.", startup_id)


def segment_transcripts(raw_dir: str | Path = "data/raw", output_path: str | Path = "data/interim/segments.parquet") -> pd.DataFrame:
    """Read raw transcripts, segment them, and write data/interim/segments.parquet."""
    raw_path = Path(raw_dir)
    out_path = Path(output_path)
    rows: list[dict[str, object]] = []
    files_by_batch_startup: dict[tuple[int, str], set[str]] = defaultdict(set)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw transcript directory not found: {raw_path}")

    records = collect_transcript_files(raw_path)
    canonical_ids = canonical_startup_ids(records)

    for record in records:
        path = record.path
        batch = record.batch
        startup_id = canonical_ids[record.key]
        role_policy = record.role_policy
        source = record.source
        files_by_batch_startup[(batch, startup_id)].add(source)
        parse_html = batch == 3 and "html_updated" in path.name.lower()
        text = read_transcript(path, parse_html=parse_html)
        rows.extend(make_segments(startup_id, batch, role_policy, source, text))

    warn_missing_feedback(files_by_batch_startup)

    df = pd.DataFrame(rows, columns=["startup_id", "batch", "role", "source", "text"])
    if not df.empty:
        counters: dict[tuple[str, str], int] = defaultdict(int)
        segment_ids = []
        for row in df.itertuples(index=False):
            key = (row.startup_id, row.role)
            counters[key] += 1
            segment_ids.append(f"{row.startup_id}_{row.role}_{counters[key]}")
        df.insert(4, "segment_id", segment_ids)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print the required segmentation summary tables."""
    if df.empty:
        print("No segments created.")
        return

    startup_counts = (
        df[["batch", "startup_id"]]
        .drop_duplicates()
        .groupby("batch")
        .size()
        .rename("startup_count")
        .reset_index()
    )
    segment_counts = (
        df.groupby(["role", "source"])
        .size()
        .rename("segment_count")
        .reset_index()
        .sort_values(["role", "source"])
    )

    print("Startup count per batch:")
    print(startup_counts.to_string(index=False))
    print("\nSegment counts by role and source:")
    print(segment_counts.to_string(index=False))


def run_segmentation(
    raw_dir: str | Path = "data/raw",
    output_path: str | Path = "data/interim/segments.parquet",
) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    df = segment_transcripts(raw_dir=raw_dir, output_path=output_path)
    print_summary(df)
    print(f"\nWrote {len(df):,} segments to {output_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment Campus Founders transcripts.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output", default="data/interim/segments.parquet")
    args = parser.parse_args()
    run_segmentation(raw_dir=args.raw_dir, output_path=args.output)


if __name__ == "__main__":
    main()
