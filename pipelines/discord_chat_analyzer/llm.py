import logging
import tiktoken
import argparse
import asyncio

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .infra import paging_message_content
from .infra import MessageRepository

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# ===========================================================
# CONFIGURATION
# ===========================================================


CONFIG__TOKENS_THRESHOLD = 4000


def get_openai_llm():
    return ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key=SecretStr(""),
        temperature=0.3,
    )


TOKENIZER = tiktoken.get_encoding("cl100k_base")


def token_count(text: str) -> int:
    return len(TOKENIZER.encode(text))


async def main():
    message_repository = MessageRepository()

    await message_repository.get_messages_content(limit=200, offset=0)


# ===========================================================
# TESTS
# ===========================================================
import pytest


pytestmark = pytest.mark.asyncio


def count_tokens_length(messages: list[str]):
    return sum([token_count(msg) for msg in messages])


def split_input_by_threshold(messages: list[str]) -> list[list[str]]:
    tokens = count_tokens_length(messages)
    if tokens < CONFIG__TOKENS_THRESHOLD:
        return [
            messages,
        ]

    messages_divided = (tokens // CONFIG__TOKENS_THRESHOLD) + 1
    step = len(messages) // messages_divided
    partitions = [messages[i : i + step] for i in range(0, len(messages), step)]
    logging.info(f"Tokens: {tokens}, Partitions: {len(partitions)}, aligned: {[len(part) for part in partitions]}")
    return partitions


@pytest.mark.target
async def test_pipeline():
    message_repository = MessageRepository()
    async for page, page_length in paging_message_content(message_repository, 200):
        partitions = split_input_by_threshold([msg.content for msg in page])


async def test_splitting_messages():
    message_repository = MessageRepository()
    page, length = await anext(paging_message_content(message_repository, 200))
    partitions = split_input_by_threshold([msg.content for msg in page])
    logging.info(f"Length each partition: {partitions}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    asyncio.run(main())
