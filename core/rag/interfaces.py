from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ISearchRag(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Hybrid search.

        Args:
            query (str): Human-readable search query.
            top_k (int): Number of top results.

        Returns:
            List[Dict[str, Any]]: List of results with fields:
                - "product_name"
                - "product_description"
                - "technical_specs"
                - "manufacturer"
                - "score" (float, от 0 до 1)
        """
        ...

    def upload_database(self) -> str:
        """
        Upload database to Qdrant.

        Returns:
            str: Status message.
        """
        ...

    def get_manufacturers(self) -> list[str]:
        """
        Return list of unique manufacturers.

        Returns:
            List[str]: List of manufacturers.
        """
        ...
