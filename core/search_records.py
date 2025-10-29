# ===========================================================
# prepare & facilities
# ===========================================================

import logging
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Generator

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.qdrant_client import QdrantClient

from core.lib.env_config import EnvConfig

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARN)  # for direct running


CONFIG = EnvConfig()
QDRANT_COLLECTION = "search_records-products"
DATABASE_PATH = Path("./data/products.db")

assert DATABASE_PATH.exists(), f"{DATABASE_PATH} does not exist"


def op_delete():
    with sqlite3.connect(DATABASE_PATH) as conn:
        res = conn.execute("SELECT * FROM products WHERE Deleted = '1'")
        records = res.fetchall()
        logging.debug(f"records: {records}")


@lru_cache
def get_lms_embedding_model():
    embedding_model = lms.list_loaded_models("embedding")
    if embedding_model is None or len(embedding_model) == 0:
        embedding_model = lms.embedding_model(CONFIG.EMBEDDING_MODEL)
        logging.info(embedding_model)

    # check_embedding_ctx_length=False - required for LMStudio
    return OpenAIEmbeddings(base_url=CONFIG.OPENAI_URL, check_embedding_ctx_length=False)


@lru_cache
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(CONFIG.QDRANT_URL)


def get_vectorestore(embedding: OpenAIEmbeddings, embedding_dimensions: int) -> QdrantVectorStore:
    client = get_qdrant_client()
    logging.debug(f"{embedding_dimensions=}")
    if client.get_collection(collection_name=QDRANT_COLLECTION) is None:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=types.VectorParams(size=embedding_dimensions, distance=Distance.COSINE),
        )
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION,
        field_name="manufacturer",
        field_schema=PayloadSchemaType.TEXT,
    )
    return QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION, embedding=embedding)


from datetime import datetime

date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ===========================================================
# core
# ===========================================================

import lmstudio as lms
from qdrant_client.conversions import common_types as types
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PayloadSchemaType, PointStruct


def sql_to_qdrant_point(sql_record) -> PointStruct:
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
            "date_modified": date_str,
        }
        return PointStruct(
            id=record[0],
            vector=summary_embedding,
            payload=raw_document,
        )

    summarized_sql_record = _summarize_sql_record(sql_record)
    cleaned_record_summary = _clean_record_summary(summarized_sql_record)
    record_summary_embedding = get_lms_embedding_model().embed_query(cleaned_record_summary)
    qdrant_point = _prepare_for_qdrant(sql_record, cleaned_record_summary, record_summary_embedding)
    return qdrant_point


def db_iterator(batch_size: int = 100) -> Generator:
    with sqlite3.connect(DATABASE_PATH) as conn:
        # headers = conn.execute("PRAGMA table_info(Products)").fetchall()
        # headers = [header[1] for header in headers]
        # logging.info(f"{headers=}")

        op_sql = conn.execute("SELECT * FROM Products")
        total_records_transformed = 0

        while True:
            if not (rows := op_sql.fetchmany(batch_size)):
                return

            yield rows

            total_records_transformed += len(rows)
            logging.info(f"{total_records_transformed=}")


def database_upload(vs, client):
    _DELETE_MARK = -1

    for batch in db_iterator(batch_size=100):
        points: list[PointStruct] = []
        for record in batch:
            if record[_DELETE_MARK]:
                try:
                    vs.delete(documents=[{"id": str(record[0])}])
                except Exception:
                    points.append(sql_to_qdrant_point(record))
                continue
            else:
                print(f"Summarizing record with ID {record[0]}")
                points.append(sql_to_qdrant_point(record))

        client.upsert(QDRANT_COLLECTION, points)


def main():
    vs_client = get_qdrant_client()
    embedding_model = get_lms_embedding_model()
    vs = get_vectorestore(embedding_model, CONFIG.EMBEDDING_MODEL_DIMENSIONS)
    database_upload(vs, vs_client)

    # user_input = input("Text for search: ")
    user_input = "Dubious parenting advice"
    vector = embedding_model.embed_query(user_input)
    res = vs.similarity_search_by_vector(vector, k=5)
    for r in res:
        print(r.page_content, r.metadata)

    def _sort_results(results, field_name, descending=False):
        return sorted(results, key=lambda x: x.payload.get(field_name, ""), reverse=descending)

    user_input = "tennis racket"
    vector = embedding_model.embed_query(user_input)
    search_filter = Filter(
        must=[
            FieldCondition(
                key="manufacturer",
                match=MatchValue(value="Banana Angel inc."),
            )
        ]
    )
    res = vs_client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vector,
        query_filter=search_filter,
        limit=5,
        with_payload=["product_name"],
    )
    sorted_res = _sort_results(res.points, "product_name", descending=False)
    for r in sorted_res:
        print(r)


if __name__ == "__main__":
    main()

# ===========================================================
# tests
# ===========================================================

import pytest


@pytest.mark.facilities
def test_db_op_delete():
    op_delete()


def test_vectorestore_with_embeddings():
    vs = get_vectorestore(get_lms_embedding_model(), CONFIG.EMBEDDING_MODEL_DIMENSIONS)
    vs.add_texts(["hello", "world"], ids=[1, 2])
    res = vs.search("hello", search_type="similarity", k=1)
    assert len(res) == 1


def test_database_upload():
    client = get_qdrant_client()
    embedding = get_lms_embedding_model()
    vs = get_vectorestore(embedding, CONFIG.EMBEDDING_MODEL_DIMENSIONS)
    database_upload(vs, client)
