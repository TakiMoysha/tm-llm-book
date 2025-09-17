import os
import yaml
from typing import Iterator, override
import pytest
from pathlib import Path

from langchain_community.document_loaders.obsidian import ObsidianLoader

import logging

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


@pytest.fixture(autouse=True)
def vault_path() -> Path:
    _vault_path = os.getenv("VAULT_PATH", None)

    if _vault_path is None or not os.path.exists(_vault_path):
        raise Exception(f"VAULT_PATH is inavlid: {_vault_path}")

    return Path(_vault_path)


from langchain_core.documents import Document
from langchain_community.document_loaders import ObsidianLoader


class TMObsidianVaultLoader(ObsidianLoader):
    @override
    def lazy_load(self) -> Iterator[Document]:
        paths = list(Path(self.file_path).glob("Base/*.md"))
        for path in paths:
            with open(path, encoding=self.encoding) as f:
                text = f.read()

            logging.info(f"{path=}")
            front_matter = self._parse_front_matter(text)
            logging.info(f"{front_matter=}")
            tags = self._parse_document_tags(text)
            dataview_fields = self._parse_dataview_fields(text)
            text = self._remove_front_matter(text)
            metadata = {
                "source": str(path.name),
                "path": str(path),
                "created": path.stat().st_ctime,
                "last_modified": path.stat().st_mtime,
                "last_accessed": path.stat().st_atime,
                **self._to_langchain_compatible_metadata(front_matter),
                **dataview_fields,
            }

            if tags or front_matter.get("tags"):
                metadata["tags"] = ",".join(tags | set(front_matter.get("tags", []) or []))
                # logging.info(f"metadata: {metadata}")

            yield Document(page_content=text, metadata=metadata)


@pytest.mark.skip("vault problem")
@pytest.mark.parametrize("edge_yaml", ['tags:\n  - backend\n  - \ncreated:\n  - "202307241245"'])
def test_should_load_none(edge_yaml: str):
    may_be_none = yaml.safe_load(edge_yaml)
    assert any([x for x in may_be_none if x is None]), f"yaml is contain a None: {may_be_none}"


def test_obsidian_vault_load(vault_path: Path):
    loader = ObsidianLoader(vault_path)
    vault = loader.load()
    assert True
