import enum
import logging

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
# ===========================================================
# tooling
# ===========================================================

import pathlib

# root of project folder
ROOT_DIR = pathlib.Path(__file__).parent


def get_asset(filename: str) -> pathlib.Path:
    if isinstance(filename, pathlib.Path):
        if not pathlib.Path(filename).is_file():
            raise ValueError(f"File does not exist: {filename}")

        path = filename

    elif isinstance(filename, str):
        path = ROOT_DIR / "assets" / filename

    return path


class DataSets(enum.Enum):
    dense_vector = "query.public.10K.fbin"
    sparse_vector = ""


class FBinReader:
    """
    Sync+Async iterator over *.fbin file
    """


# ===========================================================
# elastic-/open- search
# ===========================================================

# ===========================================================
# pgvector
# ===========================================================

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


@pytest.mark.target
def test_target(): ...
