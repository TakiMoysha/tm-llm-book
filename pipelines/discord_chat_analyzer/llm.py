import argparse
import asyncio

from .infra import MessageRepository


async def main():
    message_repository = MessageRepository()
    await message_repository.try_create_table()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.parse_args()

    asyncio.run(main())
