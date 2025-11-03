# ===========================================================
# prepare & facilities
# ===========================================================

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Generator

import qdrant_client
import qdrant_client.http.exceptions
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.qdrant_client import QdrantClient
from tabulate import tabulate

from core.lib.env_config import EnvConfig
from core.rag.interfaces import ISearchRag, UIHooks
from core.rag.query_context import QueryContext

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARN)  # for direct running


CONFIG = EnvConfig()
QDRANT_COLLECTION = "search_records-products"
DATABASE_PATH = Path("./data/products.db")

assert DATABASE_PATH.exists(), f"{DATABASE_PATH} does not exist"


class ProductsLoader:
    _path: Path

    def __init__(self, path: Path | str):
        if isinstance(path, str):
            path = Path(path)

        if not path.exists() or not path.is_file():
            raise ValueError(f"{path} does not exist or is not a file")

        self._path = path

    def debug_call(self):
        with sqlite3.connect(DATABASE_PATH) as conn:
            res = conn.execute("SELECT * FROM products")
            records = res.fetchall()
            logging.debug(f"DatabaseLoader.debug_call:{records=}")

    def into_iter(self, batch_size: int = 100) -> Generator:
        with sqlite3.connect(str(self._path)) as conn:
            # headers = conn.execute("PRAGMA table_info(Products)").fetchall()
            # headers = [header[1] for header in headers]
            # logging.debug(f"DB: {headers=}")

            op_sql = conn.execute("SELECT * FROM Products")

            while rows := op_sql.fetchmany(batch_size):
                yield rows


@dataclass
class EmbeddingContext:
    model: OpenAIEmbeddings
    dimensions: int


@lru_cache
def get_lms_embedding_model_context() -> EmbeddingContext:
    embedding_model = lms.list_loaded_models("embedding")
    if embedding_model is None or len(embedding_model) == 0:
        embedding_model = lms.embedding_model(CONFIG.EMBEDDING_MODEL)
        logging.info(embedding_model)

    # check_embedding_ctx_length=False - required for LMStudio
    return EmbeddingContext(
        OpenAIEmbeddings(base_url=CONFIG.OPENAI_URL, check_embedding_ctx_length=False),
        CONFIG.EMBEDDING_MODEL_DIMENSIONS,
    )


@lru_cache
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(CONFIG.QDRANT_URL)


def get_vector_store(embedding_ctx: EmbeddingContext) -> QdrantVectorStore:
    client = get_qdrant_client()
    try:
        client.get_collection(collection_name=QDRANT_COLLECTION)
    except qdrant_client.http.exceptions.UnexpectedResponse as err:
        if err.status_code != 404:
            raise err

        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=types.VectorParams(size=embedding_ctx.dimensions, distance=Distance.COSINE),
        )

    client.create_payload_index(
        collection_name=QDRANT_COLLECTION,
        field_name="manufacturer",
        field_schema=PayloadSchemaType.TEXT,
    )
    return QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION, embedding=embedding_ctx.model)




# ===========================================================
# core
# ===========================================================

import lmstudio as lms
from qdrant_client.conversions import common_types as types
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PayloadSchemaType, PointStruct


def sql_to_qdrant_point(sql_record) -> PointStruct:
    _date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _summarize_sql_record(sql_record):
        record_summary_string = f"""
        Product_Name: {sql_record[1]}
        Product Description: {sql_record[2]}
        Technical Specs: {sql_record[3]}
        Manufacturer: {sql_record[4]}"""
        logging.info(f"Summarized record for {sql_record[1]}: {record_summary_string}")
        return record_summary_string

    def _clean_record_summary(record_summary: str):
        return record_summary.replace("--", "")

    def _prepare_for_qdrant(record, summary, summary_embedding):
        raw_document = {
            "id": record[0],
            "summary": summary,
            "vector": summary_embedding,
            "product_name": record[1],
            "product_description": record[2],
            "technical_specs": record[3],
            "manufacturer": record[4],
            "date_modified": _date_str,
        }
        return PointStruct(
            id=record[0],
            vector=summary_embedding,
            payload=raw_document,
        )

    summarized_sql_record = _summarize_sql_record(sql_record)
    cleaned_record_summary = _clean_record_summary(summarized_sql_record)
    record_summary_embedding = get_lms_embedding_model_context().model.embed_query(cleaned_record_summary)
    qdrant_point = _prepare_for_qdrant(sql_record, cleaned_record_summary, record_summary_embedding)
    return qdrant_point


class QdrantSearchRag(ISearchRag):
    embedding_ctx: EmbeddingContext
    _qdrant_client: QdrantClient | None = None
    _vector_store: QdrantVectorStore | None = None

    def __init__(self, embedding_ctx: EmbeddingContext | None = None):
        self.embedding_ctx = embedding_ctx or get_lms_embedding_model_context()

    @property
    def embedding_model(self):
        if self.embedding_ctx is None:
            self.embedding_ctx = get_lms_embedding_model_context()

        return self.embedding_ctx.model

    @property
    def qdrant_client(self):
        if self._qdrant_client is None:
            self._qdrant_client = get_qdrant_client()

        return self._qdrant_client

    @property
    def vector_store(self):
        if self._vector_store is None:
            self._vector_store = get_vector_store(self.embedding_ctx)

        return self._vector_store

    def load_sources(
        self,
        *,
        batch_size: int = 100,
        ui_hooks: UIHooks | None = None,
    ):
        _DELETE_MARK_INDEX = -1

        _progress_update_cb = getattr(ui_hooks, "on_progress_update", (lambda _: None))

        products_source = ProductsLoader(DATABASE_PATH)

        for i, batch in enumerate(products_source.into_iter(batch_size=batch_size)):
            points: list[PointStruct] = []

            for record in batch:
                should_upsert = True
                try:
                    if record[_DELETE_MARK_INDEX] and self.vector_store.delete(documents=[{"id": str(record[0])}]):
                        should_upsert = False
                except Exception as err:
                    logging.error(f"Failed to delete record {record}: {err}")
                    should_upsert = True

                if should_upsert:
                    points.append(sql_to_qdrant_point(record))

                _progress_update_cb(i + 1)

            self.qdrant_client.upsert(QDRANT_COLLECTION, points)

    # TODO: переписать на монойдный манер
    def search(self, query: QueryContext): ...

    def example_similarity_search(self, query: str, *, top_k: int = 5):
        """Simple example of search based on similarity."""
        vector = self.embedding_model.embed_query(query)
        return self.vector_store.similarity_search_by_vector(vector, k=top_k)

    def example_search_with_filter(self, query: str, *, top_k: int = 5):
        """
        Example with filter and sorting.

        Sorting include in 'search' because results can returned in different order,
        so it's better to sort them as part of procedure.
        """
        vector = self.embedding_model.embed_query(query)
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="manufacturer",
                    match=MatchValue(value="Banana Angel inc."),
                )
            ]
        )
        res = self.qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vector,
            query_filter=search_filter,
            limit=top_k,
            with_payload=["product_name"],
        )

        def _sort_results(results, field_name, descending=False):
            return sorted(results, key=lambda x: x.payload.get(field_name, ""), reverse=descending)

        return _sort_results(res.points, "product_name", descending=False)


def main():
    rag = QdrantSearchRag()
    # rag.load_sources()

    results: list[Document] = rag.example_similarity_search("Dubious parenting advice", top_k=5)
    print(tabulate(results, headers=("",)))

    results = rag.example_search_with_filter("tennis racket")
    print(tabulate(results, headers=("",)))


if __name__ == "__main__":
    main()


# ===========================================================
# tests
# ===========================================================

import pytest


@pytest.fixture(name="db")
def _database():
    return ProductsLoader(DATABASE_PATH)


@pytest.fixture(name="qdrant_rag")
def qdrant_rag():
    return QdrantSearchRag()


def test_db_op_delete(db: ProductsLoader):
    db.debug_call()


def test_vector_store_with_embeddings():
    vs = get_vector_store(get_lms_embedding_model_context())
    vs.add_texts(["hello", "world"], ids=[1, 2])
    res = vs.search("hello", search_type="similarity", k=1)
    assert len(res) == 1


@pytest.mark.integration
def test_database_upload(rag: QdrantSearchRag):
    rag.load_sources()
