import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import TokenTextSplitter
from pydantic import BaseModel, Field, SecretStr

WORKING_DIR = Path(__file__).parent


# ===========================================================
# preprocessing
# ===========================================================

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


def _parse_timestamp(timestamp_str: str) -> float:
    parts = timestamp_str.strip().split(":")
    hours, minutes, seconds = int(parts[0]), int(parts[1]), float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


@dataclass()
class VTTSegment:
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
    segments: dict[TSegmentTimeIndex, VTTSegment] = {}

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
                segments[segment_index] = VTTSegment()

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
# LLM Basics
# ===========================================================
import lmstudio as lms
from langchain_openai import ChatOpenAI


def lms_load_model():
    model_name = "ibm/granite-4-h-tiny"
    # model_name = "mistralai/mistral-nemo-instruct-2407"
    model = lms.llm(model_name, config={})
    # class Segment(BaseModel):
    #     title: str
    #     content: str
    # model.respond(response_format=Segment)
    return model


HARDWARE_MODEL = None


def load_chat_model():
    global HARDWARE_MODEL
    HARDWARE_MODEL = lms_load_model()

    llm: ChatOpenAI = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key=SecretStr(""),
        temperature=0,
        max_tokens=8000,
    )
    return llm


# ===========================================================
# hierarchical segmentation
# ===========================================================
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate


class SummarySegment(BaseModel):
    title: str = Field()
    content: str = Field()


class SummarySegmentList(BaseModel):
    segments: list[SummarySegment]


parser = PydanticOutputParser(pydantic_object=SummarySegmentList)

split_prompt = PromptTemplate.from_template("""
You are a lecture structure analyzer. Split the following transcript into logical, thematic segments.
Each segment should have a clear title and contain a coherent idea.

Return ONLY a valid JSON object with structure:
{format_instructions}

Transcript:
{transcript}
""").partial(format_instructions=parser.get_format_instructions())

summarize_prompt = PromptTemplate.from_template("""
You are an academic summarizer. Summarize the following lecture segment in clear, concise English.  
Preserve key terms, definitions, and logical structure. Do not add interpretations.  
Use Markdown with bullet points if needed.

Segment title: {title}
Segment content:
{content}

Summary:
""")

aggregate_prompt = PromptTemplate.from_template("""
You are a lecture synthesizer. Combine the following summaries of lecture segments into one coherent, comprehensive summary.  
Maintain logical flow, preserve terminology, and ensure no important concept is lost.  
Use Markdown with headings for each major section.

Summaries:
{summaries}

Final comprehensive summary:
""")

# ==========================================================================================


def split_text_by_token(text, *, max_tokens: int = 6000) -> list[str]:
    splitter = TokenTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=200,
        encoding_name="cl100k_base",
    )
    chunks = splitter.split_text(text)
    return chunks


def segmentation_processing(chain, text) -> str: ...


def run_summarize_pipeline(text):
    llm = load_chat_model()

    chunks = split_text_by_token(text)

    all_segments = []

    for i, chunk in enumerate(chunks):
        chain = split_prompt | llm | parser

        try:
            raw_segment = chain.invoke({"transcript": chunk}, config={})
            all_segments.extend(raw_segment.segments)
        except Exception as err:
            logging.error(f"Failed to decode CHUNK{i + 1}: {err}", exc_info=True)
            all_segments.append({"title": f"Segment {i + 1}_fallback", "content": chunk})

        # try:
        # segment = json.loads(raw_segment)
        # debug_segments.append(segment)
        # print(segment)
        # all_segments.extend(segment)
        # except json.JSONDecodeError as e:
        #     logging.error(f"Failed to decode JSON: {e}", exc_info=True)
        #     all_segments.append({"title": f"Segment {i + 1}_fallback", "content": chunk})

        logging.info(f"Segment {i + 1} done")

    print(json.dumps(all_segments, indent=4))

    with open("segments.json", "w") as f:
        json.dump(all_segments, f, indent=4)


# segment_chain = summarize_prompt | llm | StrOutputParser()
# summaries = []
# for segment in raw_segments:
#     summary = segment_chain.invoke({"title": segment["title"], "content": segment["content"]})
#     summaries.append(summary)
#
# aggregate_chain = aggregate_prompt | llm | StrOutputParser()
# final_summary = aggregate_chain.invoke({"summaries": "\n".join(summaries)})

# print(final_summary)


if __name__ == "__main__":
    text = (WORKING_DIR / "./data/test_subs.en.vtt.md").read_text()

    run_summarize_pipeline(text)

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
