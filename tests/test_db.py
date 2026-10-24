import pytest
from qdrant_client import QdrantClient

from src.db.qdrant import QdrantCatalogClient
from src.pipeline.enrichment import (
    enrich_catalog_item,
)


@pytest.fixture
def in_memory_client() -> QdrantCatalogClient:
    """Fixture providing an isolated in-memory QdrantCatalogClient."""
    qc = QdrantClient(":memory:")
    catalog = QdrantCatalogClient(client=qc, collection_name="test_fashion_catalog", embedding_dim=4)
    catalog.init_collection(recreate=True)
    return catalog


def test_collection_initialization(in_memory_client: QdrantCatalogClient):
    """Verifies collection creation, idempotency, and recreation."""
    assert in_memory_client.client.collection_exists(in_memory_client.collection_name)
    assert in_memory_client.count() == 0

    # Idempotent call without recreate
    created = in_memory_client.init_collection(recreate=False)
    assert created is False

    # Recreation call
    recreated = in_memory_client.init_collection(recreate=True)
    assert recreated is True
    assert in_memory_client.client.collection_exists(in_memory_client.collection_name)


def test_deterministic_enrichment():
    """Verifies that enrichment produces identical, valid values across runs."""
    sample_1 = {
        "id": 15970,
        "gender": "Men",
        "masterCategory": "Apparel",
        "subCategory": "Topwear",
        "articleType": "Shirts",
        "baseColour": "Navy Blue",
        "season": "Fall",
        "year": 2011.0,
        "usage": "Casual",
        "productDisplayName": "Turtle Check Men Navy Blue Shirt",
    }
    sample_2 = dict(sample_1)

    item1 = enrich_catalog_item(sample_1)
    item2 = enrich_catalog_item(sample_2)

    # Determinism
    assert item1.id == item2.id
    assert item1.price == item2.price
    assert item1.rating == item2.rating
    assert item1.brand == item2.brand
    assert item1.in_stock == item2.in_stock
    assert item1.tags == item2.tags

    # Content checks
    assert item1.brand == "Turtle"
    assert 19.99 <= item1.price <= 149.99
    assert 3.2 <= item1.rating <= 5.0
    assert "navy" in item1.tags
    assert "shirts" in item1.tags
    assert "men" in item1.tags


def test_enrichment_field_fallbacks():
    """Verifies that missing optional fields fallback gracefully."""
    minimal = {"id": 99999}
    item = enrich_catalog_item(minimal)
    assert item.id == 99999
    assert item.master_category == "Apparel"
    assert item.gender == "Unisex"
    assert item.price > 0.0
    assert 1.0 <= item.rating <= 5.0
    assert isinstance(item.in_stock, bool)
    assert isinstance(item.brand, str) and len(item.brand) > 0


def test_batch_upsert_and_count(in_memory_client: QdrantCatalogClient):
    """Verifies batch upserting and count reporting."""
    items = [
        {
            "id": 1,
            "product_display_name": "Nike Running Shoes",
            "master_category": "Footwear",
            "gender": "Men",
            "price": 120.0,
            "rating": 4.5,
            "in_stock": True,
            "tags": ["shoes", "running", "nike"],
        },
        {
            "id": 2,
            "product_display_name": "Puma Casual T-shirt",
            "master_category": "Apparel",
            "gender": "Women",
            "price": 35.0,
            "rating": 4.0,
            "in_stock": False,
            "tags": ["apparel", "casual", "puma"],
        },
    ]
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]

    in_memory_client.upsert_catalog_items(items, embeddings)
    assert in_memory_client.count() == 2

    stats = in_memory_client.get_stats()
    assert stats["exists"] is True
    assert stats["points_count"] == 2


def test_get_by_id(in_memory_client: QdrantCatalogClient):
    """Verifies retrieving a specific product by ID."""
    items = [
        {"id": 101, "product_display_name": "Titan Watch", "price": 89.99},
    ]
    embeddings = [[0.5, 0.5, 0.5, 0.5]]
    in_memory_client.upsert_catalog_items(items, embeddings)

    found = in_memory_client.get_by_id(101)
    assert found is not None
    assert found["product_display_name"] == "Titan Watch"
    assert found["price"] == 89.99

    assert in_memory_client.get_by_id(999) is None


def test_vector_similarity_search(in_memory_client: QdrantCatalogClient):
    """Verifies nearest-neighbor search ranks by cosine similarity."""
    items = [
        {"id": 1, "product_display_name": "Item A", "price": 20.0},
        {"id": 2, "product_display_name": "Item B", "price": 40.0},
    ]
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    in_memory_client.upsert_catalog_items(items, embeddings)

    # Query vector very close to Item A
    query_vec = [0.99, 0.01, 0.0, 0.0]
    results = in_memory_client.search(query_vec, limit=2)

    assert len(results) == 2
    assert results[0].id == 1
    assert results[0].score > 0.95
    assert results[1].id == 2


def test_hybrid_filtering_price_range(in_memory_client: QdrantCatalogClient):
    """Verifies hybrid vector search with scalar price range conditions."""
    items = [
        {"id": 1, "product_display_name": "Cheap Shirt", "price": 25.0},
        {"id": 2, "product_display_name": "Medium Shirt", "price": 45.0},
        {"id": 3, "product_display_name": "Expensive Shirt", "price": 120.0},
    ]
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
        [0.8, 0.2, 0.0, 0.0],
    ]
    in_memory_client.upsert_catalog_items(items, embeddings)

    # Query with max_price = 50.0
    f = in_memory_client.build_filter(max_price=50.0)
    results = in_memory_client.search([1.0, 0.0, 0.0, 0.0], filter_conditions=f)

    returned_ids = [r.id for r in results]
    assert 1 in returned_ids
    assert 2 in returned_ids
    assert 3 not in returned_ids
    for r in results:
        assert r.payload["price"] <= 50.0


def test_hybrid_filtering_multi_condition(in_memory_client: QdrantCatalogClient):
    """Verifies multi-constraint boolean filters (gender, category, in_stock)."""
    items = [
        {
            "id": 1,
            "product_display_name": "Men Polo",
            "gender": "Men",
            "master_category": "Apparel",
            "in_stock": True,
            "rating": 4.5,
            "price": 30.0,
        },
        {
            "id": 2,
            "product_display_name": "Women Dress",
            "gender": "Women",
            "master_category": "Apparel",
            "in_stock": True,
            "rating": 4.8,
            "price": 75.0,
        },
        {
            "id": 3,
            "product_display_name": "Men Jeans (Out of stock)",
            "gender": "Men",
            "master_category": "Apparel",
            "in_stock": False,
            "rating": 4.2,
            "price": 55.0,
        },
    ]
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
    ]
    in_memory_client.upsert_catalog_items(items, embeddings)

    # Filter: Men AND in_stock AND max_price <= 40
    f = in_memory_client.build_filter(
        gender="Men",
        master_category="Apparel",
        in_stock_only=True,
        max_price=40.0,
    )
    results = in_memory_client.search([1.0, 0.0, 0.0, 0.0], filter_conditions=f)

    assert len(results) == 1
    assert results[0].id == 1
    assert results[0].payload["gender"] == "Men"
    assert results[0].payload["in_stock"] is True
    assert results[0].payload["price"] <= 40.0


def test_hybrid_filtering_color(in_memory_client: QdrantCatalogClient):
    """Verifies base_colour filtering in QdrantCatalogClient."""
    items = [
        {"id": 1, "product_display_name": "Nike Black Shoe", "brand": "Nike", "base_colour": "Black"},
        {"id": 2, "product_display_name": "Nike White Shoe", "brand": "Nike", "base_colour": "White"},
    ]
    embeddings = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    in_memory_client.upsert_catalog_items(items, embeddings)

    f = in_memory_client.build_filter(brand="Nike", base_colour="Black")
    results = in_memory_client.search([1.0, 0.0, 0.0, 0.0], filter_conditions=f)
    assert len(results) == 1
    assert results[0].id == 1
    assert results[0].payload["base_colour"] == "Black"


def test_hybrid_filtering_tags(in_memory_client: QdrantCatalogClient):
    """Verifies keyword tag matching."""
    items = [
        {"id": 1, "product_display_name": "Sneakers", "tags": ["footwear", "running", "sports"]},
        {"id": 2, "product_display_name": "Formal Shoes", "tags": ["footwear", "formal", "office"]},
    ]
    embeddings = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    in_memory_client.upsert_catalog_items(items, embeddings)

    f = in_memory_client.build_filter(tags=["formal"])
    results = in_memory_client.search([0.0, 1.0, 0.0, 0.0], filter_conditions=f)
    assert len(results) == 1
    assert results[0].id == 2


def test_ingest_catalog_pipeline():
    """Verifies end-to-end ingestion runner on a small slice with MobileCLIP2 embeddings."""
    from src.pipeline.ingestion import ingest_catalog

    mem_client = QdrantCatalogClient(
        client=QdrantClient(":memory:"),
        collection_name="test_ingest_catalog",
        embedding_dim=512,
    )

    stats = ingest_catalog(
        limit=4,
        batch_size=2,
        recreate=True,
        client=mem_client,
    )

    assert stats["items_indexed"] == 4
    assert stats["collection_stats"]["points_count"] == 4
    assert mem_client.count() == 4

    # Verify we can query an item by its actual catalog ID
    points, _ = mem_client.client.scroll(mem_client.collection_name, limit=1)
    assert len(points) == 1
    first_item = mem_client.get_by_id(points[0].id)
    assert first_item is not None
    assert "product_display_name" in first_item

