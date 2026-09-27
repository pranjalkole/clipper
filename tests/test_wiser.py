import math
from unittest.mock import MagicMock

import pytest
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, ScoredPoint

from src.db.qdrant import QdrantCatalogClient
from src.models.schemas import ComposedSearchRequest, WISERConfig
from src.models.wiser import (
    WISERComposedRetriever,
    WISERComposer,
    WISEREditor,
    WISERRefiner,
    WISERVerifier,
    cosine_similarity,
    l2_normalize,
)


def test_l2_normalize_and_cosine_similarity():
    """Validates vector normalization and cosine similarity calculation."""
    vec = [3.0, 4.0, 0.0]
    normalized = l2_normalize(vec)
    assert pytest.approx(math.sqrt(sum(x * x for x in normalized)), abs=1e-5) == 1.0
    assert pytest.approx(normalized[0], abs=1e-5) == 0.6
    assert pytest.approx(normalized[1], abs=1e-5) == 0.8

    # Cosine similarity between identical unit vectors is 1.0
    assert pytest.approx(cosine_similarity(normalized, normalized), abs=1e-5) == 1.0

    # Orthogonal vectors similarity is 0.0
    vec_b = [0.0, 0.0, 1.0]
    assert pytest.approx(cosine_similarity(normalized, vec_b), abs=1e-5) == 0.0


def test_wiser_composer_vector_math():
    """Validates MobileCLIP2 feature composition norm(v_img + alpha*v_mod)."""
    v_img = [1.0, 0.0, 0.0, 0.0]
    v_mod = [0.0, 1.0, 0.0, 0.0]

    # Equal weights alpha=1.0 -> [1/sqrt(2), 1/sqrt(2), 0, 0]
    v_comp = WISERComposer.compose_embeddings(v_img, v_mod, alpha=1.0)
    norm = math.sqrt(sum(x * x for x in v_comp))
    assert pytest.approx(norm, abs=1e-5) == 1.0
    assert pytest.approx(v_comp[0], abs=1e-5) == 1.0 / math.sqrt(2)
    assert pytest.approx(v_comp[1], abs=1e-5) == 1.0 / math.sqrt(2)

    # Higher alpha shifts composition toward modification
    v_comp_high = WISERComposer.compose_embeddings(v_img, v_mod, alpha=2.0)
    assert v_comp_high[1] > v_comp_high[0]

    # Dimension mismatch
    with pytest.raises(ValueError, match="Dimension mismatch"):
        WISERComposer.compose_embeddings([1.0, 0.0], [1.0, 0.0, 0.0])


def test_wiser_editor_deterministic_fallback():
    """Validates rule-based caption generation in offline mode."""
    context = {
        "master_category": "Apparel",
        "article_type": "Shirts",
        "gender": "Men",
        "brand": "Roadster",
    }
    caption = WISEREditor.generate_target_caption(
        modification_text="same in solid black with collar",
        catalog_context=context,
        llm=None,
    )
    assert "Men" in caption
    assert "Roadster" in caption
    assert "Shirts" in caption
    assert "solid black with collar" in caption


def test_wiser_editor_llm_invocation():
    """Validates target caption generation when LLM is provided."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "A classic men's slim fit cotton polo t-shirt in solid black."
    mock_llm.invoke.return_value = mock_response

    test_img = Image.new("RGB", (64, 64), color="blue")
    caption = WISEREditor.generate_target_caption(
        modification_text="make it black polo shirt",
        reference_image=test_img,
        llm=mock_llm,
    )
    assert caption == "A classic men's slim fit cotton polo t-shirt in solid black."
    mock_llm.invoke.assert_called_once()


def test_wiser_verifier_adaptive_fusion():
    """Validates adaptive fusion and confidence gate calculation across candidates."""
    v_t2i = [0.0, 1.0, 0.0, 0.0]
    v_comp = [0.7071, 0.7071, 0.0, 0.0]
    v_mod = [0.0, 1.0, 0.0, 0.0]
    v_img = [1.0, 0.0, 0.0, 0.0]

    # Candidate 1: Highly aligned with modification (T2I & mod)
    pt1 = ScoredPoint(
        id=1,
        version=1,
        score=0.95,
        payload={"id": 1, "product_display_name": "Target Match Shirt", "brand": "BrandA"},
        vector=[0.1, 0.99, 0.0, 0.0],
    )
    # Candidate 2: Aligned only with reference image (misses modification)
    pt2 = ScoredPoint(
        id=2,
        version=1,
        score=0.70,
        payload={"id": 2, "product_display_name": "Old Style Shirt", "brand": "BrandB"},
        vector=[0.99, 0.1, 0.0, 0.0],
    )

    cfg = WISERConfig(w_t2i=0.45, w_comp=0.35, w_mod=0.20, confidence_threshold=0.70)
    result = WISERVerifier.verify_and_fuse(
        candidate_points=[pt1, pt2],
        v_t2i=v_t2i,
        v_comp=v_comp,
        v_mod=v_mod,
        v_img=v_img,
        config=cfg,
    )

    assert result.pool_size == 2
    assert len(result.candidates) == 2
    # Candidate 1 should rank first due to higher modification and T2I alignment
    assert result.candidates[0].id == 1
    assert result.candidates[0].rank == 1
    assert result.candidates[0].score > result.candidates[1].score
    assert result.candidates[0].wiser_t2i_score is not None
    assert result.top_confidence > 0.70
    assert result.verified is True


def test_wiser_refiner_fallback():
    """Validates refiner discrepancy diagnosis in fallback mode."""
    refinement = WISERRefiner.diagnose_and_refine(
        top_candidates=[],
        modification_text="red running shoes",
        current_target_caption="running shoes",
        llm=None,
    )
    assert "red running shoes" in refinement.discrepancy_analysis
    assert "red running shoes" in refinement.refined_target_caption


def test_wiser_composed_retriever_end_to_end():
    """Validates complete Retrieve-Verify-Refine pipeline using in-memory Qdrant database."""
    qc = QdrantClient(":memory:")
    catalog = QdrantCatalogClient(client=qc, collection_name="test_wiser_catalog", embedding_dim=4)
    catalog.init_collection(recreate=True)

    # Ingest 3 catalog items
    items = [
        {
            "id": 101,
            "product_display_name": "Navy Blue Polo T-Shirt",
            "master_category": "Apparel",
            "sub_category": "Topwear",
            "gender": "Men",
            "price": 39.99,
            "rating": 4.6,
            "in_stock": True,
        },
        {
            "id": 102,
            "product_display_name": "Red Graphic T-Shirt",
            "master_category": "Apparel",
            "sub_category": "Topwear",
            "gender": "Men",
            "price": 24.99,
            "rating": 4.2,
            "in_stock": True,
        },
        {
            "id": 103,
            "product_display_name": "Black Formal Shoes",
            "master_category": "Footwear",
            "sub_category": "Shoes",
            "gender": "Men",
            "price": 89.99,
            "rating": 4.8,
            "in_stock": True,
        },
    ]
    # Unit vectors
    embeddings = [
        [0.0, 1.0, 0.0, 0.0],  # Item 101: Navy polo
        [0.5, 0.5, 0.5, 0.5],  # Item 102: Graphic t-shirt
        [0.0, 0.0, 1.0, 0.0],  # Item 103: Shoes
    ]
    catalog.upsert_catalog_items(items, embeddings)

    # Mock embedder returning 4-dimensional unit vectors
    mock_embedder = MagicMock()
    mock_embedder.embed_text.side_effect = lambda t: [0.0, 1.0, 0.0, 0.0]
    mock_embedder.embed_image.side_effect = lambda img: [1.0, 0.0, 0.0, 0.0]

    retriever = WISERComposedRetriever(
        embedder=mock_embedder,
        client=catalog,
        llm=None,
        config=WISERConfig(confidence_threshold=0.60),
    )

    req = ComposedSearchRequest(
        modification_text="same style in navy blue",
        limit=2,
    )

    resp = retriever.retrieve(req)
    assert resp.total_found >= 1
    assert resp.candidates[0].id == 101
    assert resp.candidates[0].score > 0.60
    assert resp.verified is True
    assert resp.latency_ms > 0
