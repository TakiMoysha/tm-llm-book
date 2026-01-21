import os
import logging
import time
import tiktoken
import argparse
import asyncio
from typing import List

from langchain_openai import ChatOpenAI
from pydantic import SecretStr, BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from pipelines.draft.text_summarizer import summarize_prompt

from .infra import paging_message_content
from .infra import MessageRepository

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# ===========================================================
# CONFIGURATION
# ===========================================================

CONFIG__OPENAI_URL = os.getenv("OPENAI_URL", "http://localhost:1234/v1")
CONFIG__OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

CONFIG__TOKENS_THRESHOLD = 4000


def get_openai_llm():
    return ChatOpenAI(
        base_url=CONFIG__OPENAI_URL,
        api_key=SecretStr(CONFIG__OPENAI_API_KEY),
        temperature=0.3,
    )


TOKENIZER = tiktoken.get_encoding("cl100k_base")


# ===========================================================
# MAIN
# ===========================================================
def token_count(text: str) -> int:
    return len(TOKENIZER.encode(text))


# ==========================================================================================

class AnalysisResult(BaseModel):
    user_cases: List[str] = Field(description="Реальные примеры, где пользователи делятся своими наработками (модами)")
    questions: List[str] = Field(description="Конкретные технические или концептуальные вопросы, заданные пользователями")
    topics: List[str] = Field(description="Повторяющиеся темы или области интересов в сообщениях")

analyzer_parser = PydanticOutputParser(pydantic_object=AnalysisResult)

analyze_prompt = PromptTemplate.from_template("""
Ты — эксперт по анализу сообщений сообщества разработчиков модов для игры Hytale.
Твоя задача — проанализировать список сообщений игроков и выделить:
1) User Cases — реальные примеры, где пользователи делятся своими наработками (например: 'Я создал мод, который добавляет ...'),
2) Вопросы — конкретные технические или концептуальные вопросы, заданные пользователями,
3) Темы — ключевые области интересов, которые повторяются в сообщениях (например: 'анимация', 'API', 'оптимизация').
Ты можешь обобщать и переформулировать вопросы для упрощения и/или суммаризации.
Не включай общие фразы вроде 'привет' или 'спасибо'.
Игнорируй сообщения, не связанные с разработкой модов. "

Return ONLY a valid JSON object with structure:
{format_instructions}

Messages:
{messages}
""").partial(format_instructions=analyzer_parser.get_format_instructions())

def analyze_questions(messages: list[str]) -> dict:
    chat = get_openai_llm()
    chain = analyze_prompt | chat | analyzer_parser
    result = chain.invoke({"messages": messages})
    return result

summarize_prompt = PromptTemplate.from_template(""""
""")


async def main():
    message_repository = MessageRepository()

    await message_repository.get_messages_content(limit=200, offset=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    asyncio.run(main())

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
        return [ messages ]

    messages_divided = (tokens // CONFIG__TOKENS_THRESHOLD) + 1
    step = len(messages) // messages_divided
    partitions = [messages[i : i + step] for i in range(0, len(messages), step)]
    logging.info(f"Tokens: {tokens}, Partitions: {len(partitions)}, aligned: {[len(part) for part in partitions]}")
    return partitions


@pytest.mark.target
@pytest.mark.timeout(0)
async def test_analyze_pipeline():
    message_repository = MessageRepository()
    async for page, page_length in paging_message_content(message_repository, 10):
        partitions = split_input_by_threshold([msg.content for msg in page])

        for part in partitions:
            start_time = time.perf_counter()
            answer = analyze_questions(partitions[0])
            logging.info(f"[{time.perf_counter() - start_time}] Answer: {answer}")


async def test_splitting_messages():
    message_repository = MessageRepository()
    page, length = await anext(paging_message_content(message_repository, 200))
    partitions = split_input_by_threshold([msg.content for msg in page])
    logging.info(f"Length each partition: {partitions}")

