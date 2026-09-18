import logging
import threading
from pathlib import Path
from types import TracebackType
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Range,
    ScoredPoint,
    VectorParams,
)
from typing_extensions import Self

from src.core.config import settings

logger = logging.getLogger(__name__)


class QdrantCatalogClient:
    """
    Production wrapper around Qdrant vector database for fashion catalog indexing and retrieval.
    Supports in-memory mode, local embedded storage, or remote cluster instances.
    """

    def __init__(
        self,
        client: QdrantClient | None = None,
        collection_name: str | None = None,
        embedding_dim: int | None = None,
    ):
        self.collection_name = collection_name or settings.collection_name
        self.embedding_dim = embedding_dim or settings.embedding_dim

        if client is not None:
            self.client = client
        elif settings.qdrant_url:
            logger.info("Connecting to remote Qdrant via URL: %s", settings.qdrant_url)
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
        elif settings.qdrant_path:
            if settings.qdrant_path == ":memory:":
                logger.info("Initializing in-memory Qdrant instance")
                self.client = QdrantClient(":memory:")
            else:
                storage_path = Path(settings.qdrant_path)
                storage_path.mkdir(parents=True, exist_ok=True)
                logger.info("Initializing local disk Qdrant at %s", storage_path)
                self.client = QdrantClient(path=str(storage_path))
        else:
            logger.info(
                "Connecting to Qdrant at %s:%d", settings.qdrant_host, settings.qdrant_port
            )
            self.client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                api_key=settings.qdrant_api_key,
            )

    def init_collection(
        self,
        recreate: bool = False,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 100,
    ) -> bool:
        """
        Initializes collection with 512d Cosine distance metric, HNSW index, and payload schemas.

        Args:
            recreate: If True, deletes existing collection if present before creating anew.
            hnsw_m: Number of edges per node in HNSW graph.
            hnsw_ef_construct: Number of neighbours to consider during HNSW graph construction.

        Returns:
            True if collection was created, False if collection already existed and was kept.
        """
        exists = self.client.collection_exists(self.collection_name)
        if exists:
            if recreate:
                logger.warning("Recreating collection '%s' (deleting existing)", self.collection_name)
                self.client.delete_collection(self.collection_name)
            else:
                logger.info("Collection '%s' already exists. Skipping creation.", self.collection_name)
                return False

        logger.info(
            "Creating collection '%s' (dim=%d, distance=Cosine, HNSW m=%d, ef=%d)",
            self.collection_name,
            self.embedding_dim,
            hnsw_m,
            hnsw_ef_construct,
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=hnsw_m, ef_construct=hnsw_ef_construct),
        )

        # Setup payload indexes for high-performance faceted filtering
        payload_schemas = [
            ("master_category", PayloadSchemaType.KEYWORD),
            ("sub_category", PayloadSchemaType.KEYWORD),
            ("article_type", PayloadSchemaType.KEYWORD),
            ("brand", PayloadSchemaType.KEYWORD),
            ("gender", PayloadSchemaType.KEYWORD),
            ("season", PayloadSchemaType.KEYWORD),
            ("price", PayloadSchemaType.FLOAT),
            ("rating", PayloadSchemaType.FLOAT),
            ("in_stock", PayloadSchemaType.BOOL),
            ("tags", PayloadSchemaType.KEYWORD),
        ]

        for field_name, schema in payload_schemas:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not create index for '%s' (%s): %s", field_name, schema, exc
                )

        logger.info("Collection '%s' initialized with payload indexes.", self.collection_name)
        return True

    def upsert_points(self, points: list[PointStruct], wait: bool = True) -> None:
        """
        Upserts a list of PointStruct points into the collection.
        """
        if not points:
            return
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=wait,
        )

    def upsert_catalog_items(
        self,
        items: list[dict[str, Any]],
        embeddings: list[list[float]],
        wait: bool = True,
    ) -> None:
        """
        Builds PointStruct instances from item payloads and embedding vectors, then upserts.
        """
        if len(items) != len(embeddings):
            raise ValueError(
                f"Mismatch between items count ({len(items)}) and embeddings count ({len(embeddings)})"
            )

        points = [
            PointStruct(
                id=int(item["id"]),
                vector=embeddings[idx],
                payload=item,
            )
            for idx, item in enumerate(items)
        ]
        self.upsert_points(points, wait=wait)

    @staticmethod
    def build_filter(
        master_category: str | list[str] | None = None,
        sub_category: str | list[str] | None = None,
        article_type: str | list[str] | None = None,
        gender: str | list[str] | None = None,
        brand: str | list[str] | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        min_rating: float | None = None,
        in_stock_only: bool | None = None,
        tags: list[str] | None = None,
    ) -> Filter | None:
        """
        Constructs a Qdrant Filter object combining category, brand, price, rating,
        stock, and tag constraints.
        """
        conditions: list[FieldCondition] = []

        def _add_match_condition(key: str, value: str | list[str] | None) -> None:
            if value is None:
                return
            if isinstance(value, list):
                clean_list = [str(v).strip() for v in value if str(v).strip()]
                if clean_list:
                    conditions.append(
                        FieldCondition(key=key, match=MatchAny(any=clean_list))
                    )
            elif str(value).strip():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=str(value).strip()))
                )

        _add_match_condition("master_category", master_category)
        _add_match_condition("sub_category", sub_category)
        _add_match_condition("article_type", article_type)
        _add_match_condition("gender", gender)
        _add_match_condition("brand", brand)

        # Price range filter
        if min_price is not None or max_price is not None:
            conditions.append(
                FieldCondition(
                    key="price",
                    range=Range(gte=min_price, lte=max_price),
                )
            )

        # Rating range filter
        if min_rating is not None:
            conditions.append(
                FieldCondition(
                    key="rating",
                    range=Range(gte=min_rating),
                )
            )

        # In-stock filter
        if in_stock_only is True:
            conditions.append(
                FieldCondition(
                    key="in_stock",
                    match=MatchValue(value=True),
                )
            )

        # Tags match
        if tags:
            clean_tags = [str(t).lower().strip() for t in tags if str(t).strip()]
            if clean_tags:
                conditions.append(
                    FieldCondition(
                        key="tags",
                        match=MatchAny(any=clean_tags),
                    )
                )

        if not conditions:
            return None

        return Filter(must=conditions)

    def search(
        self,
        query_vector: list[float],
        filter_conditions: Filter | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
    ) -> list[ScoredPoint]:
        """
        Executes vector nearest neighbor search with optional faceted boolean filtering.

        Args:
            query_vector: 512d query embedding.
            filter_conditions: Optional Qdrant Filter.
            limit: Maximum number of points to retrieve.
            score_threshold: Minimum cosine similarity score.

        Returns:
            List of ScoredPoint instances ranked by similarity.
        """
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=filter_conditions,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        )
        return response.points

    def get_by_id(self, product_id: int) -> dict[str, Any] | None:
        """
        Retrieves product payload by numeric product ID.
        """
        records = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[product_id],
            with_payload=True,
            with_vectors=False,
        )
        if records and records[0].payload:
            return records[0].payload
        return None

    def count(self) -> int:
        """Returns total number of points in collection."""
        if not self.client.collection_exists(self.collection_name):
            return 0
        return self.client.count(self.collection_name).count

    def get_stats(self) -> dict[str, Any]:
        """Returns collection metadata and statistics."""
        if not self.client.collection_exists(self.collection_name):
            return {"exists": False, "points_count": 0}
        info = self.client.get_collection(self.collection_name)
        return {
            "exists": True,
            "status": str(info.status),
            "points_count": info.points_count,
            "vectors_count": getattr(info, "vectors_count", None),
            "segments_count": info.segments_count,
        }

    def close(self) -> None:
        """Closes the underlying Qdrant client connection."""
        try:
            self.client.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Error while closing Qdrant client: %s", exc)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()


# Thread-safe singleton instance
_catalog_client: QdrantCatalogClient | None = None
_client_lock = threading.Lock()


def get_qdrant_client() -> QdrantCatalogClient:
    """Returns thread-safe singleton QdrantCatalogClient instance."""
    global _catalog_client
    if _catalog_client is None:
        with _client_lock:
            if _catalog_client is None:
                _catalog_client = QdrantCatalogClient()
    return _catalog_client


def reset_qdrant_client() -> None:
    """Resets singleton QdrantCatalogClient instance (primarily for testing)."""
    global _catalog_client
    with _client_lock:
        _catalog_client = None
