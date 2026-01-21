"""
Implementation of summarization of large materials, I used hierarchical summarization.

: TODO: divide pipelines into - ingestor, segmenter, summarizer, aggregator, exporter
: TODO: add recursive aggregation to compress text (divide-and-conquer, divide-and-merge).
: TODO: async/parallel processing
: - TODO: AbstractLLMProvider (on package level)
"""

import logging
import tiktoken
from argparse import ArgumentParser
from typing import cast

import pydantic
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# ==========================================================================================

CONFIG__SEGMENT_TOKEN_SIZE = 4000
CONFIG__SEGMENT_OVERLAP = 500

CONFIG__RECURSIVE_AGGREGATION_MAX_SIZE = 12000
CONFIG__RECURSIVE_AGGREGATION_BLOCK_SIZE = 2

TOKENIZER = tiktoken.get_encoding("cl100k_base")

# ==========================================================================================
import lmstudio as lms

HARDWARE_MODEL = None


def lms_load_model():
    model_name = "qwen/qwen3-4b-2507"
    global HARDWARE_MODEL
    HARDWARE_MODEL = lms.llm(model_name, config={})
    return HARDWARE_MODEL


def get_openai_llm():
    return ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key=SecretStr(""),
        temperature=0.1,
    )


# ===========================================================
# hierarchical segmentation
# ===========================================================
import tiktoken
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


class SummarySegment(pydantic.BaseModel):
    title: str = pydantic.Field()
    content: str = pydantic.Field()


class SummarySegmentList(pydantic.BaseModel):
    segments: list[SummarySegment]


splitter_parser = PydanticOutputParser(pydantic_object=SummarySegmentList)

split_prompt = PromptTemplate.from_template("""
You are a lecture structure analyzer. Split the following transcript into logical, thematic segments.
Each segment should have a clear title and contain a coherent idea.

Return ONLY a valid JSON object with structure:
{format_instructions}

Transcript:
{transcript}
""").partial(format_instructions=splitter_parser.get_format_instructions())

summarize_prompt = PromptTemplate.from_template("""
You are an academic summarizer. Summarize the following lecture segment in clear, concise English.  
Leave only important semantic information - peculiarities of speech, politeness, etiquette, etc. - it doesn’t matter.
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



def token_count(text: str) -> int:
    return len(TOKENIZER.encode(text))


def split_input(
    text: str,
    *,
    max_tokens: int = CONFIG__SEGMENT_TOKEN_SIZE,
    overlap: int = CONFIG__SEGMENT_OVERLAP,
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=overlap,
        length_function=token_count,
        is_separator_regex=False,
    )
    chunks = splitter.split_text(text)
    logging.info(f"Split into {len(chunks)} chunks")
    return chunks


# ===========================================================
# RAG actions
# : added DI for get_openai_llm
# ===========================================================
def segmentate(input_text: str, **kwargs):
    chat = get_openai_llm()
    chunks = split_input(input_text)
    all_segments = []
    chain = split_prompt | chat | splitter_parser

    for i, chunk in enumerate(chunks):
        result = chain.invoke({"transcript": chunk})
        all_segments.extend(result.segments)

    return all_segments


def summarize(segments: list[SummarySegment], **kwargs):
    chat = get_openai_llm()
    chain = summarize_prompt | chat | StrOutputParser()
    return [chain.invoke({"title": segment.title, "content": segment.content}) for segment in segments]


def aggregate(summaries: list[str], **kwargs):
    chat = get_openai_llm()
    chain = aggregate_prompt | chat
    return chain.invoke({"summaries": "\n".join(summaries)})


def recursive_aggregate(
    summaries: list[str],
    *,
    max_tokens: int = CONFIG__RECURSIVE_AGGREGATION_MAX_SIZE,
    block_size: int = CONFIG__RECURSIVE_AGGREGATION_BLOCK_SIZE,
    depth: int = 0,
    **kwargs,
) -> str:
    if len(summaries) == 1:
        return summaries[0]

    llm = get_openai_llm()
    total_tokens = token_count("\n".join(summaries))
    summary_chain = aggregate_prompt | llm | StrOutputParser()
    aggregation_chain = aggregate_prompt | llm | StrOutputParser()

    if total_tokens < max_tokens:
        result = aggregation_chain.invoke({"summaries": "\n".join(summaries)})
        logging.debug(f"Depth {depth}: {result}")
        return result

    blocks: list[list[str]] = []
    for i in range(0, len(summaries), block_size):
        blocks.append(summaries[i : i + block_size])

    logging.debug(f"Depth {depth}: Split into {len(blocks)} blocks")

    new_summaries: list[str] = []
    for idx, block in enumerate(blocks):
        summary = summary_chain.invoke({"summaries": "\n".join(block)})
        logging.debug(f"Depth {depth}: {summary}")
        new_summaries.append(summary)

    logging.debug(f"Depth {depth}: Aggregated into {len(new_summaries)} summaries")

    return recursive_aggregate(
        summaries=new_summaries,
        max_tokens=max_tokens,
        block_size=block_size,
        depth=depth + 1,
    )


# ==========================================================================================


def hierarchical_summarize_pipeline(input_text: str, **kwargs):
    text_token_size = len(TOKENIZER.encode(input_text))
    logging.info(f"Input text token size: {text_token_size}")

    segments = segmentate(input_text)
    logging.info(f"Segmented into {len(segments)} segments")

    summaries = cast(list[str], summarize(segments))
    logging.info(f"Summarized into {len(summaries)} summaries")

    final_summary = aggregate(summaries)
    return final_summary


def main(input: str, **kwargs):
    with open(input, "rb") as f:
        text = f.read().decode("utf-8")

    summary = hierarchical_summarize_pipeline(text)

    with open("summary-output.txt", "w") as f:
        f.write(str(summary))


if __name__ == "__main__":
    stdio_parser = ArgumentParser(
        prog="SSTConverter",
        description="Speech-to-Text converter with langchain and OpenAI interface.",
    )

    stdio_parser.add_argument(
        "input",
        type=str,
        help="Path to input audio files: mp3, wav.",
    )
    stdio_parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        default="output.txt",
        help="Path to output text file, as raw text file.",
    )

    support_group = stdio_parser.add_argument_group("support")
    support_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG level for logging.",
    )
    support_group.add_argument(
        "--load-model",
        action="store_true",
        help="Load model into memory.",
    )

    args = vars(stdio_parser.parse_args())

    if args.get("debug") and logging.getLogger().setLevel(logging.DEBUG):
        logging.debug("DEBUG logging enabled")

    if args.get("load_model"):
        llm = lms_load_model()
        logging.debug(f"Loaded model: {HARDWARE_MODEL}")

    main(**args)
