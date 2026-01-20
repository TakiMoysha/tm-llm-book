import asyncio
import collections
import datetime
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Iterable, TypedDict, cast

import aiosqlite
import dateutil
import httpx
from lib.interfaces import IRepository
from tqdm.auto import tqdm

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

DB_NAME = "discord_analytics.db"

# ==========================================================================================


@dataclass
class Config:
    discord_token: str | None = field(default_factory=lambda: os.getenv("DISCORD_TOKEN"), repr=False)

    def __post_init__(self):
        if self.discord_token is None:
            raise ValueError("DISCORD_TOKEN is not set")


# ==========================================================================================


class MessageRepository:
    def __init__(self, url: str = DB_NAME, *, batch_size: int = 1000) -> None:
        self._url = url
        self._batch_size = batch_size

    async def try_create_table(self):
        SQL_CREATE_MESSAGES_TABLE = """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                channel_id TEXT,
                author_id TEXT,
                author_username TEXT,
                content TEXT,
                message_json JSON
            );
        """
        async with aiosqlite.connect(self._url) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(SQL_CREATE_MESSAGES_TABLE)
                await cursor.execute("PRAGMA synchronous = NORMAL;")
                await conn.commit()

    def _fs_cache(self, buffer: list):
        with open(f"{datetime.datetime.now().timestamp()}.messages.json", "w") as f:
            json.dump(buffer, f)

    async def bulk_save(self, messages: list[dict]):
        SQL_INSERT_MESSAGE = """
            INSERT INTO messages (id, channel_id, author_id, author_username, content, message_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO NOTHING
        """

        # self._fs_cache(messages)
        if not messages:
            logging.warning("No messages to save")
            return

        total = len(messages)
        saved = 0
        failed_chunks = 0
        initial_changes = 0
        _start_time = time.perf_counter()

        with tqdm(total=total, desc="Saving messages", unit="msg") as pbar:
            async with aiosqlite.connect(self._url) as conn:
                for i in range(0, total, self._batch_size):
                    _start_chunk_time = time.perf_counter()
                    chunk = messages[i : i + self._batch_size]
                    chunk_data = [
                        (
                            message["id"],
                            message["channel_id"],
                            message["author"]["id"],
                            message["author"]["username"],
                            message["content"],
                            json.dumps(message),
                        )
                        for message in chunk
                    ]

                    try:
                        await conn.executemany(SQL_INSERT_MESSAGE, chunk_data)
                        saved += conn.total_changes - initial_changes  # added only no duplicate keys
                        initial_changes = conn.total_changes
                        pbar.update(len(chunk))
                    except Exception as err:
                        failed_chunks += 1
                        logging.error(f"Unexpected error in chunk {i // self._batch_size}: {err}")

                    pbar.update(len(chunk))

                    _end_chunk_time = time.perf_counter()
                    pbar.set_description(f"Saved {saved}/{total} in {_end_chunk_time - _start_time:.2f} seconds")

                await conn.commit()

        _end_time = time.perf_counter()
        logging.info(f"Saved {saved}/{total} in {_end_time - _start_time:.2f} seconds. Failed chunks: {failed_chunks}")

    async def get_messages_content(self, limit: int = 200, offset: int = 0):
        MessageContent = collections.namedtuple("MessageContent", ["cursor", "content"])
        async with aiosqlite.connect(self._url) as conn:
            conn.row_factory = MessageContent
            res_cur = await conn.execute(
                "SELECT content FROM messages LIMIT ? OFFSET ?",
                (limit, offset),
            )
            result, length = cast(Iterable[MessageContent], await res_cur.fetchall()), res_cur.rowcount
            return result, length


assert isinstance(MessageRepository, IRepository)


class DiscordAdapter:
    _repo: IRepository

    def __init__(self, repo: IRepository, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._repo = repo

    async def get_messages_per_day(self, channel_id: str, *, limit: int = 100, headers: dict[str, str] | None = None):
        DISCORD_API_MESSAGES_URL = f"https://discord.com/api/channels/{channel_id}/messages"
        buffer = []
        target_last_datetime = datetime.datetime.now() - datetime.timedelta(days=1)

        def process_messages(res, _buffer):
            res.raise_for_status()
            data = res.json()
            _buffer.extend(data)
            return None if not data else data[-1]

        async with httpx.AsyncClient() as client:
            response = await client.get(
                DISCORD_API_MESSAGES_URL,
                headers=headers,
                params={"limit": limit},
            )
            support_msg = process_messages(response, buffer)

            if support_msg is None:
                return

            last_msg_datetime = dateutil.parser.parse(support_msg["timestamp"])

            while last_msg_datetime.timestamp() > target_last_datetime.timestamp():
                response = await client.get(
                    DISCORD_API_MESSAGES_URL,
                    headers=headers,
                    params={"limit": limit, "before": support_msg["id"]},
                )
                support_msg = process_messages(response, buffer)
                last_msg_datetime = dateutil.parser.parse(support_msg["timestamp"])
                await asyncio.sleep(0.4)

        return buffer


# ===========================================================
# tests
# ===========================================================
import pytest

pytestmark = pytest.mark.asyncio


class DiscordOptions(TypedDict):
    server_id: str
    channel_id: str


@pytest.fixture(name="discord_opts")
def discord_test_options(request):
    server_id = request.config.getoption("--test-server-id")
    channel_id = request.config.getoption("--test-channel-id")

    if None in [server_id, channel_id]:
        pytest.skip(f"Required input: {server_id=}, {channel_id=}")

    return DiscordOptions(server_id=server_id, channel_id=channel_id)


async def test_sqlite_json():
    async with aiosqlite.connect(":memory:") as conn:
        await conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, data JSON)")
        await conn.execute("INSERT INTO messages (id, data) VALUES (?, ?)", (1, '{"foo": "bar"}'))
        row_cursor = await conn.execute("SELECT data FROM messages WHERE id = ?", (1,))
        row = await row_cursor.fetchone()
        assert row and len(row) == 1
        cursor = await conn.execute("SELECT json_valid(data) FROM messages WHERE id = ?", (1,))
        valid = await cursor.fetchone()
        assert valid


@pytest.mark.target
async def test_pagination_message_content():
    repo = MessageRepository()

    async def paging_message_content(_repo: MessageRepository):
        page_size, offset = 200, 0
        length = -1

        while length != 0:
            messages, length = await repo.get_messages_content(limit=page_size, offset=offset)
            offset += length
            yield messages

    page_generator = await anext(paging_message_content(repo))

    for msg in page_generator:
        logging.info(f"Got <{msg[1]}> messages")


@pytest.mark.pipeline
async def test_discord_download_history(discord_opts: DiscordOptions):
    config = Config()
    message_repository = MessageRepository()
    await message_repository.try_create_table()
    discord = DiscordAdapter(repo=message_repository)
    messages = await discord.get_messages_per_day(
        discord_opts.get("channel_id"),
        headers={"Authorization": f"{config.discord_token}"},
    )
    assert messages, "Downloaded messages is None"
    await message_repository.bulk_save(messages)
