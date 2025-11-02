# ===========================================================
# prepare & facilities
# ===========================================================

import logging
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Generator
from tqdm import tqdm

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
import qdrant_client
import qdrant_client.http.exceptions
from qdrant_client.qdrant_client import QdrantClient
from core.lib.env_config import EnvConfig

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARN)  # for direct running


CONFIG = EnvConfig()
QDRANT_COLLECTION = "search_records-products"
DATABASE_PATH = Path("./data/products.db")

assert DATABASE_PATH.exists(), f"{DATABASE_PATH} does not exist"


# =========================================================== 
# agents
# =========================================================== 

product_db_agent = construct_io_search_agent()
