import argparse
import asyncio

from .infra import MessageRepository


async def main():
    message_repository = MessageRepository()

    await message_repository.get_messages_content(limit=200, offset=0)


# ===========================================================
# TESTS
# ===========================================================
import pytest
import tiktoken


@pytest.mark.target
async def test_count_message_tokens():
    ...



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    asyncio.run(main())
