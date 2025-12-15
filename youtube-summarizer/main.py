from dataclasses import dataclass, field
import re
from pathlib import Path

WORKING_DIR = Path(__file__).parent


def _parse_timestamp(timestamp_str: str) -> float:
    parts = timestamp_str.strip().split(":")
    hours, minutes, seconds = int(parts[0]), int(parts[1]), float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


@dataclass()
class Segment:
    original_text: list[str] = field(default_factory=list)
    deduplicated_text: set[str] = field(default_factory=set)

    def add(self, text: str):
        self.original_text.append(text)
        self.deduplicated_text.add(text)


def _clean_vtt(text: str, *, segment_duration: int = 30):
    """
    Clean VTT subtitles and group them into segments.

    Args:
        text: Raw VTT file content
        segment_duration: Duration of each segment in seconds (default: 30)

    Returns:
        Cleaned text grouped into segments
    """
    remove_tags_pattern = re.compile(r"<[^>]+>")
    timestamp_pattern = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2}\.\d{3})")

    def _is_metadata(line):
        return line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:")

    def _is_cue(line: str):
        return "align:" in line or "position:" in line or "size:" in line

    type TSegmentTimeIndex = int
    segments: dict[TSegmentTimeIndex, Segment] = {}

    current_timestamp = None
    for line in text.splitlines():
        line = line.strip()

        if not line or _is_metadata(line):
            continue

        if timestamp_match := timestamp_pattern.search(line):
            start_time_str = timestamp_match.group(1)
            current_timestamp = _parse_timestamp(start_time_str)
            continue

        if _is_cue(line):
            continue

        clean_text = remove_tags_pattern.sub("", line).strip()

        # empty or short line (? - may by)
        if not clean_text or len(clean_text) < 3:
            continue

        # segmentation
        if current_timestamp is not None:
            segment_index = int(current_timestamp // segment_duration)

            if segment_index not in segments:
                segments[segment_index] = Segment()

            segment = segments[segment_index]

            if clean_text not in segment.deduplicated_text:
                segment.add(clean_text)

    # join segments
    result = []
    for segment_index in sorted(segments.keys()):
        # text_list, _ = segments[segment_index]
        segment = segments[segment_index]

        segment_content = " ".join(segment.deduplicated_text)

        # Add timestamp header for each segment
        start_time = segment_index * segment_duration
        end_time = (segment_index + 1) * segment_duration

        start_parts = (int(start_time // 3600), int((start_time % 3600) // 60), int(start_time % 60))
        end_parts = (int(end_time // 3600), int((end_time % 3600) // 60), int(end_time % 60))

        timestamp_header = f"[{start_parts[0]:02d}:{start_parts[1]:02d}:{start_parts[2]:02d} - {end_parts[0]:02d}:{end_parts[1]:02d}:{end_parts[2]:02d}]"

        result.append(f"{timestamp_header}\n{segment_content}")

    return "\n\n".join(result)


def convert_vtt(file: Path):
    text = file.read_text(encoding="utf-8")
    plaintext = _clean_vtt(text)

    (file.parent / f"{file.name}.md").write_text(plaintext)


# ===========================================================
# tests
# ===========================================================

import pytest


@pytest.fixture
def subtitle_path():
    test_file = WORKING_DIR / "./data/test_subs.en.vtt"
    assert test_file.exists(), "test file does not exist"
    return test_file


def test_target(subtitle_path):
    convert_vtt(subtitle_path)
    assert (subtitle_path.parent / f"{subtitle_path.name}.md").exists()


@pytest.mark.runtime
def test_relative_paths():
    path = WORKING_DIR / "./data/test_subs.en.vtt"
    assert path.exists()
