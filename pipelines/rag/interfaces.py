from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

# ===========================================================
# CORE
# ===========================================================


@runtime_checkable
class ISearchRag(Protocol):
    def search(self, query: str, top_k: int = 5, *args, **kwargs):
        """Hybrid search."""
        ...

    def load_sources(self, *args, **kwargs) -> str:
        """Upload database to Qdrant."""
        ...


# ===========================================================
# UI
# ===========================================================


@dataclass
class UIHooks:
    on_progress_update: Callable[[int], None] | None = None
    on_status_update: Callable[[str], None] | None = None
    on_error: Callable[[Exception], None] | None = None
    on_success: Callable[[str], None] | None = None
