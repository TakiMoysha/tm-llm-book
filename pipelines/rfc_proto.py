"""
# Milestones:

 [] Build simple pipeline (Focus on RFC 8259 - JSON)
 - [] warm up the database
    - [] download RFC 8259 and save into the 'cache' directory
    - [] walk through 'cache' directory and transform files into embeddings (and save them)
 - [] work with evals
 - []
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
import subprocess

# ===========================================================
# Prepare
# ===========================================================
"""
On this stage we download the RFC's and put them into a database as embeddings for further processing.

1. Download bank of RFC's
2. Upsert each RFC into the database
3. 

"""


if (OUTPUT_DIR := Path("./data/")) and not OUTPUT_DIR.exists():
    raise Exception(f"Output directory does not exist (or wrong workdir): {OUTPUT_DIR.absolute()}")


def download_rfc(rfc_id: int) -> Path:
    """Download RFC and save it to the output directory

    Args:
        rfc_id (int): RFC ID

    Returns:
        Path: Path to the downloaded RFC
    """
    rsync_url = f"rsync.rfc-editor.org::rfcs-text-only/rfc{rfc_id}.txt"
    local_file = OUTPUT_DIR / f"rfc{rfc_id}.txt"

    try:
        result = subprocess.run(
            ["rsync", "-avz", rsync_url, str(local_file)], check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        raise Exception("Cannot download RFC.") from e

    if result.returncode != 0:
        raise Exception(f"Download failed: {result.stderr}")

    return local_file


def test_download_rfc():
    path = download_rfc(8259)
    assert path.exists()


os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================
# evals
# ===========================================================
"""
# Categories:

| Category                 | Goals                                                      |
| ------------------------ | ---------------------------------------------------------- |
| 1 Поиск по факту         | Найти RFC по описанию                                      |
| 2 Переформулировка       | Проверить, понимает ли система неформальные формулировки   |
| 3 Сравнение / различие   | Проверить понимание различий между RFC                     |
| 4 Контекстное объяснение | Проверить, может ли система объяснить концепцию            |
| 5 Иерархия / зависимость | Проверить, знает ли система, какой RFC устарел или заменён |
| 6 Связанные стандарты    | Проверить, может ли система находить смежные RFC           |
| 7 Глубокое понимание     | Проверить, может ли система применить знание на практике   |
"""

"""
# Metrics:

Ground Truth - набор правильных ответов.

| Metrics           | Description                                          |
| ----------------- | ---------------------------------------------------- |
| Hit Rate          |
| Precision         |
| Recall            |
"""
# ==========================================================================================

evals: dict[str, list[dict]] = {
    "Can I write comments in JSON?": [
        {"answer": "No, comments are not allowed in JSON.", "source": "RFC 8259, Section 2"}
    ],
    "What types can I use in JSON?": [
        {
            "answer": "You can use: strings, numbers, booleans (true, false), null, arrays, and objects.",
            "source": "RFC 8259, Section 3",
        }
    ],
    "Is 'undefined' a valid JSON value?": [
        {
            "answer": "No, 'undefined' is not valid in JSON. Only null represents absence of a value.",
            "source": "RFC 8259, Section 3",
        }
    ],
    "Are keys in JSON required to be strings?": [
        {
            "answer": "Yes, all object keys must be strings in JSON, enclosed in double quotes.",
            "source": "RFC 8259, Section 4",
        }
    ],
    "Can JSON represent NaN or Infinity?": [
        {
            "answer": "No, NaN and Infinity are not valid JSON values. Numbers must follow a specific decimal format.",
            "source": "RFC 8259, Section 6",
        }
    ],
    "Is trailing comma allowed in arrays or objects?": [
        {"answer": "No, trailing commas are not allowed in JSON arrays or objects.", "source": "RFC 8259, Section 2"}
    ],
    "Can a JSON value be just a string?": [
        {"answer": 'Yes, a JSON text can be a standalone string, such as "hello".', "source": "RFC 8259, Section 2"}
    ],
    "What is the MIME type for JSON?": [
        {"answer": "The official MIME type for JSON is application/json.", "source": "RFC 8259, Section 11"}
    ],
    "Why is {name: 'Bob'} invalid JSON?": [
        {
            "answer": 'Because keys must be double-quoted strings in JSON. It should be {"name": "Bob"}.',
            "source": "RFC 8259, Section 4",
        }
    ],
    "How should I represent a missing value in JSON?": [
        {"answer": "Use 'null' to represent a missing or empty value.", "source": "RFC 8259, Section 3"}
    ],
    "Can I use single quotes in JSON strings?": [
        {
            "answer": "No, JSON requires double quotes for strings. Single quotes are not valid.",
            "source": "RFC 8259, Section 7",
        }
    ],
    "Is an empty object {} valid JSON?": [
        {"answer": "Yes, an empty object {} is valid JSON.", "source": "RFC 8259, Section 4"}
    ],
    "Explain the JSON grammar production rules": [
        {
            "answer": "JSON is built from values: strings, numbers, objects, arrays, true, false, null. "
            "An object is {string: value}, an array is [value, ...]. "
            "Strings must be in double quotes. Numbers follow C-style syntax without leading zeros.",
            "source": "RFC 8259, Sections 2, 3, 4",
        }
    ],
}


# ===========================================================
# Implementation
# ===========================================================

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

llm: ChatOpenAI = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    temperature=0,
)


def _():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer short, precise and added source references as possible, source format: 'RFC XXXX, Section N.'",
            ),
            ("user", "Question: {input}"),
            ("user", "Context: {context}"),
        ]
    )

    # chain
    rag_chain = prompt | llm | StrOutputParser()

    #
    response = rag_chain.invoke(
        {
            "input": "Can I write comments in JSON?",
            # "context": retrieved_docs,  # это metadata={"rfc": "8259", "section": "2"}
        }
    )


# ===========================================================
# Code Testing
# ===========================================================
import pytest


@dataclass(frozen=True)
class ServerConfig:
    url: str = field(default_factory=lambda: os.getenv("OPENAI_URL"))
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"), repr=False)


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL"))
    dimensions: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_MODEL_DIMENSIONS")))


@dataclass(frozen=True)
class LLMConfig:
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL"))


@pytest.mark.fixture(name="ServerConfig", autouse=True)
def _ServerConfig():
    if (_url := os.getenv("OPENAI_URL")) is None:
        pytest.fail("OPENAI_URL is not set")

    if (_api_key := os.getenv("OPENAI_API_KEY")) is None:
        pytest.fail("OPENAI_API_KEY is not set")

    return ServerConfig(url=_url, api_key=_api_key)


@pytest.mark.fixture(name="LLMConfig", autouse=True)
def _LLMConfig():
    logging.warning("support only local llm through LM Studio")
    if (_llm_model := os.getenv("LLM_MODEL")) is None:
        pytest.fail("LLM_MODEL is not set")


@pytest.mark.fixture(name="EmbeddingConfig", autouse=True)
def _EmbeddingConfig():
    if (_embedding_model := os.getenv("EMBEDDING_MODEL")) is None:
        pytest.fail("EMBEDDING_MODEL is not set")

    if (
        _embedding_model_dimensions := os.getenv("EMBEDDING_MODEL_DIMENSIONS")
    ) is None or not _embedding_model_dimensions.isdigit():
        pytest.fail("EMBEDDING_MODEL_DIMENSIONS is not set")

    return EmbeddingConfig(
        model=_embedding_model,
        dimensions=int(_embedding_model_dimensions),
    )


# ===========================================================
# evals-driven development
# ===========================================================
