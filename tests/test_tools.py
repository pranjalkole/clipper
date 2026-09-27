from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient

from src.agents.tools import create_catalog_tools
from src.db.qdrant import QdrantCatalogClient
from src.models.schemas import WISERConfig
from src.models.wiser import WISERComposedRetriever


@pytest.fixture
def mock_catalog_setup():
    """Provides an isolated in-memory Qdrant client, mock embedder, and WISER retriever."""
    qc = QdrantClient(":memory:")
    client = QdrantCatalogClient(client=qc, collection_name="test_tools_catalog", embedding_dim=4)
    client.init_collection(recreate=True)

    items = [
        {
            "id": 1,
            "product_display_name": "Nike Air Max Sneakers",
            "master_category": "Footwear",
            "sub_category": "Shoes",
            "gender": "Men",
            "brand": "Nike",
            "price": 129.99,
            "rating": 4.8,
            "in_stock": True,
            "tags": ["shoes", "running", "nike"],
        },
        {
            "id": 2,
            "product_display_name": "Puma Classic Suede",
            "master_category": "Footwear",
            "sub_category": "Shoes",
            "gender": "Men",
            "brand": "Puma",
            "price": 65.0,
            "rating": 4.3,
            "in_stock": True,
            "tags": ["shoes", "casual", "puma"],
        },
    ]
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    client.upsert_catalog_items(items, embeddings)

    mock_embedder = MagicMock()
    mock_embedder.embed_text.side_effect = lambda t: [1.0, 0.0, 0.0, 0.0]
    mock_embedder.embed_image.side_effect = lambda img: [1.0, 0.0, 0.0, 0.0]

    retriever = WISERComposedRetriever(
        embedder=mock_embedder,
        client=client,
        llm=None,
        config=WISERConfig(confidence_threshold=0.50),
    )

    tools = create_catalog_tools(client=client, embedder=mock_embedder, retriever=retriever)
    return {
        "client": client,
        "tools": {t.name: t for t in tools},
    }


def test_tool_signatures_and_schemas(mock_catalog_setup):
    """Verifies that structured tools expose valid LangChain tool schemas."""
    tools = mock_catalog_setup["tools"]
    assert "catalog_hybrid_search" in tools
    assert "catalog_composed_search" in tools
    assert "product_details_lookup" in tools

    assert tools["catalog_hybrid_search"].args_schema is not None
    assert tools["catalog_composed_search"].args_schema is not None
    assert tools["product_details_lookup"].args_schema is not None


def test_catalog_hybrid_search_text_query(mock_catalog_setup):
    """Verifies catalog_hybrid_search with text query and scalar filters."""
    tool = mock_catalog_setup["tools"]["catalog_hybrid_search"]
    results = tool.invoke(
        {"query_text": "running sneakers", "brand": "Nike", "max_price": 150.0}
    )

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["id"] == 1
    assert results[0]["brand"] == "Nike"
    assert results[0]["score"] > 0.90


def test_catalog_composed_search(mock_catalog_setup):
    """Verifies catalog_composed_search invoking WISER engine."""
    tool = mock_catalog_setup["tools"]["catalog_composed_search"]
    result = tool.invoke(
        {
            "modification_text": "same shoes but in red",
            "master_category": "Footwear",
            "limit": 2,
        }
    )

    assert isinstance(result, dict)
    assert "candidates" in result
    assert "target_caption" in result
    assert "confidence_score" in result
    assert len(result["candidates"]) >= 1


def test_product_details_lookup(mock_catalog_setup):
    """Verifies product_details_lookup retrieves product or returns error."""
    tool = mock_catalog_setup["tools"]["product_details_lookup"]

    # Existing product
    found = tool.invoke({"product_id": 1})
    assert isinstance(found, dict)
    assert found["id"] == 1
    assert found["product_display_name"] == "Nike Air Max Sneakers"

    # Non-existing product
    not_found = tool.invoke({"product_id": 9999})
    assert isinstance(not_found, dict)
    assert "error" in not_found
