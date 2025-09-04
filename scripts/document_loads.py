import os
from pathlib import Path

from langchain_community.document_loaders.obsidian import ObsidianLoader

_vault_path = os.getenv("VAUL_PATH", None)

if _vault_path is None or not os.path.exists(_vault_path):
    raise Exception(f"VAUL_PATH is inavlid: {_vault_path}")

VAULT_PATH = Path(_vault_path)

loader = ObsidianLoader(_vault_path)
vault = loader.load()
