import asyncio
import enum
import logging
import xml.etree.ElementTree as ET
from abc import ABC
from dataclasses import dataclass, field
from functools import cached_property
from typing import AsyncGenerator

import pytest

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
# ===========================================================
# tooling
# ===========================================================

import pathlib

from sentence_transformers import SentenceTransformer

SENTENCE_MODEL = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-cos-v1")
# root project dir
ROOT_DIR = pathlib.Path(__file__).parent


class Assets(enum.Enum):
    dense_vector = "query.public.10K.fbin"
    so_vector = "posts.xml"


def get_asset(asset: Assets) -> pathlib.Path:
    return (ROOT_DIR / "assets" / asset.value).resolve()


# =========================================================== so_vector dataset tools
@dataclass(frozen=True, slots=True)
class Post:
    """
    post_type_id:
        1=Question,
        2=Answer,
        3=Orphaned tag wiki,
        4=Tag wiki excerpt,
        5=Tag wiki,
        6=Moderator nomination,
        7=Wiki Placeholder,
        8=Privilege Wiki
    accepted_answer_id: Id of the accepted answer
    parent_id: only present if PostTypeId=2
    """

    id: int
    post_type_id: int
    accepted_answer_id: int | None
    parent_id: int | None
    score: int
    view_count: int | None
    body: str | None
    title: str | None
    content_license: str | None
    favorite_count: int | None
    creation_date: str | None
    last_activity_date: str | None
    last_edit_date: str | None
    last_editor_user_id: int | None
    owner_user_id: int | None
    tags: list[str] | None

    @classmethod
    def from_elem(cls, element):
        # if element.attrib.get("Title") is None:
        # logging.warning(f"Post {element.attrib['Id']} no title")

        return cls(
            id=element.attrib["Id"],
            post_type_id=element.attrib["PostTypeId"],
            accepted_answer_id=element.attrib.get("AcceptedAnswerId"),
            parent_id=element.attrib.get("ParentId"),
            score=element.attrib["Score"],
            view_count=element.attrib.get("ViewCount"),
            body=element.attrib.get("Body"),
            title=element.attrib.get("Title"),
            content_license=element.attrib.get("ContentLicense"),
            favorite_count=element.attrib.get("FavoriteCount"),
            creation_date=element.attrib.get("CreationDate"),
            last_activity_date=element.attrib.get("LastActivityDate"),
            last_edit_date=element.attrib.get("LastEditDate"),
            last_editor_user_id=element.attrib.get("LastEditorUserId"),
            owner_user_id=element.attrib.get("OwnerUserId"),
            tags=element.attrib.get("Tags"),
        )


async def xml_file_reader(file: pathlib.Path, chunk_size: int = 1000) -> AsyncGenerator[list[Post], None]:
    tree = ET.iterparse(file, events=("start", "end"))

    posts: list[Post] = []

    for event, elem in tree:
        if event == "end" and elem.tag == "row":
            post = Post.from_elem(elem)
            posts.append(post)

            # Если достигли chunk_size, выдаём chunk
            if len(posts) >= chunk_size:
                yield posts
                posts = []

    if posts:
        yield posts


@pytest.mark.support
async def test_dataset_so_vector_iterator():
    post_gen = xml_file_reader(get_asset(Assets.so_vector))

    async for c in post_gen:
        assert 0 < len(c) <= 1000


# ===========================================================
# elastic-/open- search
# ===========================================================

# from elasticsearch import AsyncElasticsearch
#
#
# @dataclass(frozen=True)
# class ElasticSearchConfig:
#     host: str = field(default="localhost")
#     port: int = field(default=9200)
#
#     @property
#     def client(self):
#         return AsyncElasticsearch(
#             hosts=[{"host": self.host, "port": self.port}],
#         )


from opensearchpy import AsyncOpenSearch


@dataclass(frozen=True)
class OpenSearchConfig:
    host: str = field(default="localhost")
    port: int = field(default=9200)

    @cached_property
    def client(self):
        return AsyncOpenSearch(
            hosts=[{"host": self.host, "port": self.port}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_show_warn=False,
        )


# ===========================================================
# pgvector
# ===========================================================
import psycopg


@dataclass(frozen=True)
class PgVectorConfig:
    host: str = field(default="localhost")
    port: int = field(default=5432)
    db_name: str = field(default="vectordb")
    user: str = field(default="llm_user")
    password: str = field(default="llm_password")

    def into_dict(
        self,
    ):
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.db_name,
            "user": self.user,
            "password": self.password,
        }

    def __post_init__(self):
        with psycopg.connect(**self.into_dict(), autocommit=True) as conn:
            conn.execute("SELECT 1")


def init_pgvector_database():
    """
    Turn on pgvector, create test table and create index for PNN (hnsw + inner product)
    """
    config = PgVectorConfig()

    with psycopg.connect(**config.into_dict(), autocommit=True) as conn:
        conn.execute("CREATE EXTENSION vector")
        conn.execute("CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(3));")


class IVectorStore(ABC): ...


# ===========================================================
# qdrant
# ===========================================================


def run_cli(): ...


if __name__ == "__main__":
    run_cli()

# ===========================================================
# TESTS
# ===========================================================

import pytest

pytestmark = pytest.mark.asyncio


@dataclass
class OSPostSchema:
    id: int
    title: str
    body: str
    tags: list[str]
    score: int
    embedding: str

    @staticmethod
    def schema_properties():
        return {
            "title": {"type": "text"},
            "body": {"type": "text"},
            "tags": {"type": "keyword"},
            "score": {"type": "integer"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 768,
                "method": {
                    "engine": "lucene",
                    "space_type": "cosinesimil",
                    "name": "hnsw",
                    "parameters": {"ef_construction": 128, "m": 4},
                },
            },
        }


from opensearchpy import helpers


class OpenSearchStore(IVectorStore):
    def __init__(self, index_name: str = "stackoverflow_posts") -> None:
        self.config = OpenSearchConfig()
        self.index_name = index_name

    @property
    def client(self):
        return self.config.client

    async def create_index(self):
        index_mapping = {
            "settings": {
                "index": {
                    "knn": True,
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                }
            },
            "mappings": {
                "properties": OSPostSchema.schema_properties(),
            },
        }

        async with self.client as client:
            exists = await client.indices.exists(index=self.index_name)
            if exists:
                logging.info(f"Index {self.index_name} already exists, deleting...")
                await client.indices.delete(index=self.index_name)

            logging.info(f"Creating index {self.index_name}...")
            await client.indices.create(index=self.index_name, body=index_mapping)
            logging.info(f"Index {self.index_name} created successfully")

    async def bulk_insert(self, docs: list[OSPostSchema]):
        actions = [
            {
                "_index": self.index_name,
                "_id": doc.id,
                "_source": {
                    "title": doc.title,
                    "body": doc.body,
                    "tags": doc.tags,
                    "score": doc.score,
                    "embedding": doc.embedding,
                },
            }
            for doc in docs
        ]

        async with self.client as client:
            success, failed = await helpers.async_bulk(client, actions, raise_on_error=False)
            logging.info(f"Success: {success} ({len(failed) if isinstance(failed, list) else failed} failed)")
            if failed:
                logging.error(f"Failed documents: {failed[:5] if isinstance(failed, list) else failed}")


async def ingest_dataset(asset: Assets, vector_store: OpenSearchStore):
    def embed_bulk_posts(posts: list[Post]):
        titles = [post.title for post in posts if post.title is not None]
        embeddings = SENTENCE_MODEL.encode(titles, convert_to_numpy=True)

        docs = []
        for post, embedding in zip(posts, embeddings):
            docs.append(
                OSPostSchema(
                    id=post.id,
                    title=post.title,
                    body=post.body or "",
                    tags=post.tags or [],
                    score=post.score,
                    embedding=embedding.tolist(),
                )
            )
        return docs

    async for chunk in xml_file_reader(get_asset(asset)):
        docs = embed_bulk_posts(chunk)
        await vector_store.bulk_insert(docs)


@pytest.mark.target
@pytest.mark.timeout(0)
async def test_target():
    store = OpenSearchStore()
    await store.create_index()
    await ingest_dataset(Assets.so_vector, store)
