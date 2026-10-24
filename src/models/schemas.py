import base64
import io
from typing import Any, Literal
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field


class SearchFilterParams(BaseModel):
    """Faceted search filtering parameters across scalar catalog attributes."""

    model_config = ConfigDict(extra="ignore")

    master_category: str | list[str] | None = Field(
        default=None, description="Master category filter (e.g. Apparel, Footwear, Accessories)"
    )
    sub_category: str | list[str] | None = Field(
        default=None, description="Sub category filter (e.g. Topwear, Shoes, Watches)"
    )
    article_type: str | list[str] | None = Field(
        default=None, description="Article type filter (e.g. Tshirts, Casual Shoes)"
    )
    gender: str | list[str] | None = Field(
        default=None, description="Target gender filter (e.g. Men, Women, Unisex, Boys, Girls)"
    )
    base_colour: str | list[str] | None = Field(
        default=None, description="Target color filter (e.g. Black, White, Navy Blue, Red)"
    )
    brand: str | list[str] | None = Field(
        default=None, description="Brand name filter (e.g. Nike, Puma, Levi's)"
    )
    min_price: float | None = Field(default=None, ge=0.0, description="Minimum price boundary")
    max_price: float | None = Field(default=None, ge=0.0, description="Maximum price boundary")
    min_rating: float | None = Field(
        default=None, ge=1.0, le=5.0, description="Minimum customer review rating (1.0 to 5.0)"
    )
    in_stock_only: bool | None = Field(
        default=None, description="Filter for in-stock inventory only"
    )
    tags: list[str] | None = Field(
        default=None, description="Keyword tags to match (e.g. ['casual', 'summer'])"
    )


class ProductCandidate(BaseModel):
    """
    Standardized product candidate retrieved from Qdrant vector database,
    enriched with metadata and retrieval score attribution.
    """

    model_config = ConfigDict(extra="ignore")

    id: int = Field(description="Unique numeric product identifier")
    product_display_name: str = Field(description="Display title of the product")
    master_category: str = Field(default="Apparel")
    sub_category: str = Field(default="General")
    article_type: str = Field(default="Item")
    base_colour: str = Field(default="Unknown")
    gender: str = Field(default="Unisex")
    season: str = Field(default="All")
    year: int | None = Field(default=None)
    usage: str = Field(default="Casual")
    brand: str = Field(default="Generic")
    price: float = Field(default=0.0, ge=0.0, description="Product price in USD")
    rating: float = Field(default=4.0, ge=1.0, le=5.0, description="Customer review rating")
    rating_count: int = Field(default=0, ge=0, description="Number of customer reviews")
    in_stock: bool = Field(default=True, description="Inventory availability status")
    tags: list[str] = Field(default_factory=list, description="Descriptive search tags")

    # Retrieval Scoring & Attribution
    image_url: str | None = Field(
        default=None, description="Direct URL or API endpoint path to retrieve product image"
    )
    score: float = Field(
        ge=-1.0, le=1.0, description="Normalized similarity or fused confidence score"
    )
    rank: int | None = Field(default=None, description="Rank position in retrieval result")
    retrieval_path: str = Field(
        default="vector",
        description="Path through which candidate was retrieved (e.g. 't2i', 'visual_comp', 'wiser_fused')",
    )

    # WISER Attribution Scores
    wiser_t2i_score: float | None = Field(
        default=None, description="Cosine similarity to generated target caption (T2I)"
    )
    wiser_comp_score: float | None = Field(
        default=None, description="Cosine similarity to composed embedding (norm(v_img + alpha*v_mod))"
    )
    wiser_mod_score: float | None = Field(
        default=None, description="Cosine similarity to modification text direction"
    )
    wiser_vis_score: float | None = Field(
        default=None, description="Cosine similarity to reference image visual features"
    )
    match_reasons: list[str] = Field(
        default_factory=list, description="Explanatory notes on why candidate matched"
    )

    @classmethod
    def from_payload_and_score(
        cls,
        payload: dict[str, Any],
        score: float,
        rank: int | None = None,
        retrieval_path: str = "vector",
        **wiser_scores: Any,
    ) -> "ProductCandidate":
        """Instantiates ProductCandidate from a Qdrant point payload and score."""
        data = dict(payload)
        data["score"] = float(score)
        if rank is not None:
            data["rank"] = rank
        data["retrieval_path"] = retrieval_path
        if "image_url" not in data or not data.get("image_url"):
            data["image_url"] = f"/catalog/images/{data['id']}"
        for k, v in wiser_scores.items():
            if v is not None:
                data[k] = float(v)
        return cls(**data)


class WISERConfig(BaseModel):
    """
    Hyperparameters and weights for the WISER Zero-Shot Composed Image Retrieval architecture.
    """

    alpha: float = Field(
        default=1.0,
        gt=0.0,
        description="Scaling factor for modification direction in visual feature composition norm(v_img + alpha*v_mod)",
    )
    w_t2i: float = Field(
        default=0.45,
        ge=0.0,
        description="Weight for Text-to-Image (T2I) target description alignment",
    )
    w_comp: float = Field(
        default=0.35,
        ge=0.0,
        description="Weight for composed visual feature alignment (I2I)",
    )
    w_mod: float = Field(
        default=0.20,
        ge=0.0,
        description="Weight for direct modification text alignment",
    )
    confidence_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Minimum verifier confidence score required to skip self-reflection refinement",
    )
    pool_factor: int = Field(
        default=2,
        ge=1,
        description="Multiplier for initial candidate pool size (e.g. pool_factor=2 retrieves 2*limit per path)",
    )
    max_refinement_retries: int = Field(
        default=2,
        ge=0,
        description="Maximum deeper-thinking refinement attempts when confidence < threshold",
    )


class WISERVerificationResult(BaseModel):
    """Output from the WISER Verifier node evaluating candidate pool confidence."""

    candidates: list[ProductCandidate] = Field(default_factory=list)
    top_confidence: float = Field(default=0.0, description="Highest fused candidate score")
    verified: bool = Field(
        default=False, description="True if top_confidence >= confidence_threshold"
    )
    confidence_threshold: float = Field(default=0.70)
    pool_size: int = Field(default=0)


class WISERRefinement(BaseModel):
    """Output from the WISER Deeper Thinking refiner node."""

    discrepancy_analysis: str = Field(
        description="Diagnostic analysis explaining why candidate scores were uncertain"
    )
    refined_target_caption: str = Field(
        description="Reformulated target caption emphasizing unmet attributes"
    )
    suggested_filter_updates: dict[str, Any] = Field(
        default_factory=dict, description="Recommended additions or relaxations to scalar filters"
    )


class ComposedSearchRequest(BaseModel):
    """
    Request model for Composed Image Retrieval (reference image + modification text).
    """

    reference_image_base64: str | None = Field(
        default=None, description="Base64 encoded string of reference image or data URI"
    )
    reference_image_path: str | None = Field(
        default=None, description="Local file path or URL to reference image"
    )
    modification_text: str = Field(
        description="Natural language instruction modifying reference image (e.g. 'same in navy blue with collar')"
    )
    filters: SearchFilterParams | None = Field(
        default=None, description="Optional scalar metadata filters"
    )
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of candidates to return")
    wiser_config: WISERConfig = Field(
        default_factory=WISERConfig, description="WISER CIR hyperparameters"
    )

    def load_image(self) -> Image.Image | None:
        """Helper to resolve PIL.Image from base64 string or local file path."""
        if self.reference_image_base64:
            b64_str = self.reference_image_base64
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_str)
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if self.reference_image_path:
            return Image.open(self.reference_image_path).convert("RGB")
        return None


class ComposedSearchResponse(BaseModel):
    """
    Response model for Composed Image Retrieval.
    """

    modification_text: str
    target_caption: str = Field(
        description="Target descriptive caption synthesized by WISER Editor"
    )
    candidates: list[ProductCandidate] = Field(default_factory=list)
    total_found: int = 0
    confidence_score: float = 0.0
    verified: bool = False
    iterations_count: int = 1
    latency_ms: float = 0.0
    refinement_notes: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    """Public API search request for pure text, pure image, or hybrid search."""

    query_text: str | None = Field(default=None, description="Text query string")
    query_image_base64: str | None = Field(
        default=None, description="Base64 encoded query image or data URI"
    )
    query_image_path: str | None = Field(default=None, description="Local image file path")
    filters: SearchFilterParams | None = Field(
        default=None, description="Optional faceted scalar filters"
    )
    limit: int = Field(default=10, ge=1, le=100, description="Max candidates to return")
    min_confidence_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum similarity threshold"
    )


class SearchResponse(BaseModel):
    """Public API search response."""

    query: str | None = None
    candidates: list[ProductCandidate] = Field(default_factory=list)
    total_found: int = 0
    reasoning: str | None = None
    latency_ms: float = 0.0
    extracted_filters: dict[str, Any] | None = Field(
        default=None, description="Initial scalar constraints extracted from user query by router"
    )
    filters_applied: dict[str, Any] | None = Field(
        default=None, description="Active scalar constraints applied to catalog retrieval"
    )
    relaxed_filters: list[str] = Field(
        default_factory=list, description="Filters relaxed or modified during reflection loops"
    )
    reflection_notes: list[str] = Field(default_factory=list)


# LangChain Structured Tool Input Schemas
class CatalogHybridSearchInput(BaseModel):
    """Input schema for the catalog_hybrid_search LangChain tool."""

    query_text: str | None = Field(
        default=None, description="Semantic text query to search against catalog"
    )
    query_image: str | None = Field(
        default=None, description="Base64 string or file path of query image"
    )
    master_category: str | list[str] | None = Field(
        default=None, description="Category filter (e.g. Apparel, Footwear, Accessories)"
    )
    sub_category: str | list[str] | None = Field(default=None, description="Sub-category filter (e.g. Topwear, Shoes)")
    article_type: str | list[str] | None = Field(default=None, description="Article type filter")
    gender: str | list[str] | None = Field(default=None, description="Gender filter (Men, Women, Unisex, etc.)")
    base_colour: str | list[str] | None = Field(default=None, description="Color filter (Black, White, Navy Blue, Red, etc.)")
    brand: str | list[str] | None = Field(default=None, description="Brand filter (e.g. Nike, Puma, Levi's)")
    min_price: float | None = Field(default=None, ge=0.0, description="Minimum price")
    max_price: float | None = Field(default=None, ge=0.0, description="Maximum price")
    min_rating: float | None = Field(default=None, ge=1.0, le=5.0, description="Minimum rating")
    in_stock_only: bool | None = Field(default=None, description="Only in-stock products")
    limit: int = Field(default=10, ge=1, le=50, description="Number of products to retrieve")


class CatalogComposedSearchInput(BaseModel):
    """Input schema for the catalog_composed_search LangChain tool implementing WISER CIR."""

    reference_image: str | None = Field(
        default=None, description="Base64 image string or file path of the reference product image"
    )
    modification_text: str = Field(
        description="Instruction modifying the reference product (e.g. 'same cut but in navy blue and under $60')"
    )
    master_category: str | list[str] | None = Field(default=None, description="Category filter")
    sub_category: str | list[str] | None = Field(default=None, description="Subcategory filter")
    article_type: str | list[str] | None = Field(default=None, description="Article type filter")
    gender: str | list[str] | None = Field(default=None, description="Gender filter")
    base_colour: str | list[str] | None = Field(default=None, description="Color filter")
    brand: str | list[str] | None = Field(default=None, description="Brand filter")
    min_price: float | None = Field(default=None, ge=0.0, description="Minimum price")
    max_price: float | None = Field(default=None, ge=0.0, description="Maximum price")
    min_rating: float | None = Field(default=None, ge=1.0, le=5.0, description="Minimum rating")
    in_stock_only: bool | None = Field(default=None, description="In-stock only")
    limit: int = Field(default=10, ge=1, le=50, description="Max candidates to return")


class ProductDetailsLookupInput(BaseModel):
    """Input schema for looking up product details by ID."""

    product_id: int = Field(description="Unique numeric product identifier")


class AgentStateSchema(BaseModel):
    """Pydantic baseline contract for multi-agent state machines in Phase 4."""

    query_text: str | None = None
    query_image: str | None = None
    is_composed_query: bool = False
    filters: SearchFilterParams = Field(default_factory=SearchFilterParams)
    candidates: list[ProductCandidate] = Field(default_factory=list)
    confidence_score: float = 0.0
    verified: bool = False
    retry_count: int = 0
    max_retries: int = 2
    reflection_notes: list[str] = Field(default_factory=list)
    final_response: str | None = None


# Phase 5 API Request & Response Schemas
class HealthResponse(BaseModel):
    """Service health status and dependency checks."""

    status: str = Field(default="healthy", description="Service health status ('healthy' or 'degraded')")
    version: str = Field(default="0.1.0", description="Application version")
    app_name: str = Field(default="Multimodal Search Platform")
    environment: str = Field(default="development")
    qdrant_connected: bool = Field(default=True)
    embedder_loaded: bool = Field(default=True)
    uptime_seconds: float = Field(default=0.0)
    timestamp: float = Field(default=0.0)


class TextSearchRequest(BaseModel):
    """Request payload for text-to-image semantic search."""

    query_text: str = Field(description="Search query string")
    filters: SearchFilterParams | None = Field(default=None, description="Optional faceted scalar filters")
    limit: int = Field(default=10, ge=1, le=100, description="Max candidates to return")
    min_confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)


class ImageSearchRequest(BaseModel):
    """Request payload for pure visual similarity search."""

    query_image_base64: str | None = Field(default=None, description="Base64 encoded query image or data URI")
    query_image_path: str | None = Field(default=None, description="Local image path")
    filters: SearchFilterParams | None = Field(default=None, description="Optional faceted scalar filters")
    limit: int = Field(default=10, ge=1, le=100, description="Max candidates to return")


class IngestRequest(BaseModel):
    """Request parameters to trigger dataset catalog indexing."""

    limit: int | None = Field(default=100, ge=1, description="Number of items to ingest (None for all)")
    batch_size: int = Field(default=32, ge=1, le=256, description="Batch size for embedding and upsert")
    recreate: bool = Field(default=False, description="Whether to drop and recreate collection")
    streaming: bool = Field(default=False, description="Stream dataset from HF")
    dataset_name: str | None = Field(default=None, description="Hugging Face dataset identifier")
    background: bool = Field(default=False, description="Run ingestion as background task")


class IngestResponse(BaseModel):
    """Ingestion result summary."""

    status: str = Field(default="success")
    items_indexed: int = Field(default=0)
    elapsed_seconds: float = Field(default=0.0)
    throughput_items_per_sec: float = Field(default=0.0)
    collection_name: str = Field(default="fashion_catalog")
    total_points: int = Field(default=0)
    message: str | None = None


class MetricsResponse(BaseModel):
    """Real-time latency, throughput, token, and simulated cost metrics."""

    total_requests: int = Field(default=0)
    p50_latency_ms: float = Field(default=0.0)
    p90_latency_ms: float = Field(default=0.0)
    p95_latency_ms: float = Field(default=0.0)
    p99_latency_ms: float = Field(default=0.0)
    avg_latency_ms: float = Field(default=0.0)
    tokens_per_sec: float = Field(default=0.0)
    total_tokens_consumed: int = Field(default=0)
    simulated_cost_usd: float = Field(default=0.0)
    cost_per_query_usd: float = Field(default=0.0)
    uptime_seconds: float = Field(default=0.0)

