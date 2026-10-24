import base64
import io
import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool, tool
from PIL import Image

from src.core.config import settings
from src.db.qdrant import QdrantCatalogClient, get_qdrant_client
from src.models.embedder import MultimodalEmbedder, get_embedder
from src.models.schemas import (
    CatalogComposedSearchInput,
    CatalogHybridSearchInput,
    ComposedSearchRequest,
    ProductCandidate,
    ProductDetailsLookupInput,
    SearchFilterParams,
)
from src.models.wiser import WISERComposedRetriever, WISERComposer

logger = logging.getLogger(__name__)


def _load_image_from_spec(image_spec: Any) -> Image.Image | None:
    """Helper to convert various image input formats into a PIL Image."""
    if image_spec is None:
        return None
    if isinstance(image_spec, Image.Image):
        return image_spec.convert("RGB")
    if isinstance(image_spec, (str, Path)):
        spec_str = str(image_spec).strip()
        # Check if base64 data URI or raw base64
        if spec_str.startswith("data:image") or len(spec_str) > 200:
            if "," in spec_str:
                spec_str = spec_str.split(",", 1)[1]
            try:
                img_bytes = base64.b64decode(spec_str)
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as exc:
                logger.warning("Failed to decode base64 image: %s", exc)
                return None
        # File path
        path = Path(spec_str)
        if path.is_file():
            return Image.open(path).convert("RGB")
    return None


def create_catalog_tools(
    client: QdrantCatalogClient | None = None,
    embedder: MultimodalEmbedder | None = None,
    retriever: WISERComposedRetriever | None = None,
) -> list[StructuredTool]:
    """
    Factory creating LangChain structured tools with dependency injection
    for testing and LangGraph multi-agent orchestration.
    """
    catalog_client = client or get_qdrant_client()
    multimodal_embedder = embedder or get_embedder()
    composed_retriever = retriever or WISERComposedRetriever(
        embedder=multimodal_embedder, client=catalog_client
    )

    def _catalog_hybrid_search(
        query_text: str | None = None,
        query_image: str | None = None,
        master_category: str | list[str] | None = None,
        sub_category: str | list[str] | None = None,
        article_type: str | list[str] | None = None,
        gender: str | list[str] | None = None,
        base_colour: str | list[str] | None = None,
        brand: str | list[str] | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        min_rating: float | None = None,
        in_stock_only: bool | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Executes hybrid semantic search against catalog using text, image, or multimodal fusion.
        """
        if not query_text and not query_image:
            raise ValueError("At least one of query_text or query_image must be provided.")

        v_text: list[float] | None = None
        v_img: list[float] | None = None

        if query_text:
            v_text = multimodal_embedder.embed_text(query_text)
        if query_image:
            img = _load_image_from_spec(query_image)
            if img:
                v_img = multimodal_embedder.embed_image(img)

        # Vector determination
        if v_text and v_img:
            # Multimodal fusion using WISERComposer
            query_vector = WISERComposer.compose_embeddings(v_img, v_text, alpha=1.0)
        elif v_text:
            query_vector = v_text
        elif v_img:
            query_vector = v_img
        else:
            raise ValueError("Could not extract embedding from provided query.")

        qdrant_filter = catalog_client.build_filter(
            master_category=master_category,
            sub_category=sub_category,
            article_type=article_type,
            gender=gender,
            base_colour=base_colour,
            brand=brand,
            min_price=min_price,
            max_price=max_price,
            min_rating=min_rating,
            in_stock_only=in_stock_only,
        )

        scored_points = catalog_client.search(
            query_vector=query_vector,
            filter_conditions=qdrant_filter,
            limit=limit,
            with_vectors=False,
        )

        candidates = [
            ProductCandidate.from_payload_and_score(
                payload=pt.payload or {},
                score=round(float(pt.score or 0.0), 4),
                rank=idx,
                retrieval_path="hybrid_search",
            ).model_dump()
            for idx, pt in enumerate(scored_points, start=1)
        ]
        return candidates

    def _catalog_composed_search(
        modification_text: str,
        reference_image: str | None = None,
        master_category: str | list[str] | None = None,
        sub_category: str | list[str] | None = None,
        article_type: str | list[str] | None = None,
        gender: str | list[str] | None = None,
        base_colour: str | list[str] | None = None,
        brand: str | list[str] | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        min_rating: float | None = None,
        in_stock_only: bool | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """
        Executes composed image retrieval (WISER: Wider Search, Adaptive Fusion, Deeper Thinking)
        modifying a reference image with natural language.
        """
        filters = SearchFilterParams(
            master_category=master_category,
            sub_category=sub_category,
            article_type=article_type,
            gender=gender,
            base_colour=base_colour,
            brand=brand,
            min_price=min_price,
            max_price=max_price,
            min_rating=min_rating,
            in_stock_only=in_stock_only,
        )

        request = ComposedSearchRequest(
            reference_image_base64=reference_image if reference_image and len(reference_image) > 100 else None,
            reference_image_path=reference_image if reference_image and len(reference_image) <= 100 else None,
            modification_text=modification_text,
            filters=filters,
            limit=limit,
        )

        response = composed_retriever.retrieve(request)
        return response.model_dump()

    def _product_details_lookup(product_id: int) -> dict[str, Any]:
        """
        Retrieves detailed metadata for a specific product ID.
        """
        payload = catalog_client.get_by_id(product_id)
        if not payload:
            return {"error": f"Product with ID {product_id} not found in catalog."}
        return payload

    tool_hybrid = StructuredTool.from_function(
        func=_catalog_hybrid_search,
        name="catalog_hybrid_search",
        description="Search catalog using text query, image, or multimodal query with scalar filters.",
        args_schema=CatalogHybridSearchInput,
    )

    tool_composed = StructuredTool.from_function(
        func=_catalog_composed_search,
        name="catalog_composed_search",
        description=(
            "Execute WISER Composed Image Retrieval: modify a reference image with natural language "
            "instructions (e.g. 'same in navy blue with collar') with adaptive verification."
        ),
        args_schema=CatalogComposedSearchInput,
    )

    tool_lookup = StructuredTool.from_function(
        func=_product_details_lookup,
        name="product_details_lookup",
        description="Look up complete product catalog metadata by numeric product ID.",
        args_schema=ProductDetailsLookupInput,
    )

    return [tool_hybrid, tool_composed, tool_lookup]


# Module-level default tool instances using singleton clients
try:
    _default_tools = create_catalog_tools()
    catalog_hybrid_search = _default_tools[0]
    catalog_composed_search = _default_tools[1]
    product_details_lookup = _default_tools[2]
except Exception as _tool_err:  # noqa: BLE001
    logger.debug("Deferred default tools initialization due to lock or configuration: %s", _tool_err)
    catalog_hybrid_search = None  # type: ignore[assignment]
    catalog_composed_search = None  # type: ignore[assignment]
    product_details_lookup = None  # type: ignore[assignment]
