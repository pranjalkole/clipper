"""
Unit and integration tests for the automated evaluation suite and IR metrics.
"""

import json
import math
from unittest.mock import MagicMock, patch
import pytest

from src.evaluation.evaluator import (
    compute_dcg,
    compute_ndcg,
    compute_recall,
    compute_reciprocal_rank,
    compute_latency_distribution,
    evaluate_candidate_relevance,
    run_evaluation,
)


def test_compute_dcg_empty():
    """DCG of an empty list of relevance scores is 0.0."""
    assert compute_dcg([]) == 0.0


def test_compute_dcg_known_values():
    """Verifies DCG calculation against exact mathematical formula."""
    # DCG = (2^3 - 1)/log2(2) + (2^2 - 1)/log2(3) + (2^1 - 1)/log2(4)
    #     = 7 / 1 + 3 / 1.58496 + 1 / 2
    #     = 7.0 + 1.892789 + 0.5 = 9.392789
    scores = [3.0, 2.0, 1.0]
    expected = (7.0 / math.log2(2)) + (3.0 / math.log2(3)) + (1.0 / math.log2(4))
    assert math.isclose(compute_dcg(scores, k=3), expected, rel_tol=1e-5)


def test_compute_ndcg_perfect_ranking():
    """When ranking matches ideal order, NDCG must be 1.0."""
    scores = [3.0, 2.0, 1.0, 0.0]
    ideal = [3.0, 2.0, 1.0, 0.0]
    assert compute_ndcg(scores, ideal, k=4) == 1.0


def test_compute_ndcg_zero_ideal():
    """When ideal scores are all 0, NDCG should be 1.0 if retrieved are all 0, else 0.0."""
    assert compute_ndcg([0.0, 0.0], [0.0, 0.0], k=2) == 1.0
    assert compute_ndcg([1.0, 0.0], [0.0, 0.0], k=2) == 0.0


def test_compute_ndcg_suboptimal_ranking():
    """Inverted ranking produces NDCG strictly less than 1.0."""
    scores = [0.0, 1.0, 2.0, 3.0]
    ideal = [3.0, 2.0, 1.0, 0.0]
    ndcg = compute_ndcg(scores, ideal, k=4)
    assert 0.0 < ndcg < 1.0


def test_compute_recall():
    """Verifies Recall@k calculation."""
    retrieved = [101, 102, 103, 104, 105]
    relevant = {101, 103, 999}  # 3 relevant items, 2 retrieved in top 3, 2 retrieved in top 5
    assert compute_recall(retrieved, relevant, k=1) == pytest.approx(1 / 3)
    assert compute_recall(retrieved, relevant, k=3) == pytest.approx(2 / 3)
    assert compute_recall(retrieved, relevant, k=5) == pytest.approx(2 / 3)
    assert compute_recall(retrieved, set(), k=5) == 0.0


def test_compute_reciprocal_rank():
    """Verifies MRR/Reciprocal Rank calculation."""
    retrieved = [10, 20, 30, 40]
    assert compute_reciprocal_rank(retrieved, {10}, k=4) == 1.0  # rank 1 -> 1/1
    assert compute_reciprocal_rank(retrieved, {20}, k=4) == 0.5  # rank 2 -> 1/2
    assert compute_reciprocal_rank(retrieved, {30}, k=4) == pytest.approx(1 / 3)
    assert compute_reciprocal_rank(retrieved, {99}, k=4) == 0.0  # not retrieved


def test_compute_latency_distribution():
    """Verifies percentile calculations for latencies."""
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    dist = compute_latency_distribution(latencies)
    assert dist["min"] == 10.0
    assert dist["max"] == 100.0
    assert dist["mean"] == 55.0
    assert dist["p50"] == 50.0
    assert dist["p90"] == 90.0
    assert dist["p95"] == 100.0
    assert dist["p99"] == 100.0


def test_evaluate_candidate_relevance():
    """Verifies graded relevance assignments for candidates."""
    query_spec = {
        "relevant_ids": [101, 102],
        "target_criteria": {
            "master_category": "Footwear",
            "sub_category": "Shoes",
            "gender": "Men",
            "base_colour": "Black",
            "max_price": 100.0,
        },
    }

    # Exact ID match -> 3.0
    cand_exact = {"id": 101, "master_category": "Footwear"}
    assert evaluate_candidate_relevance(cand_exact, query_spec) == 3.0

    # Full criteria match -> 2.5
    cand_full = {
        "id": 999,
        "master_category": "Footwear",
        "sub_category": "Shoes",
        "gender": "Men",
        "base_colour": "Black",
        "price": 80.0,
    }
    assert evaluate_candidate_relevance(cand_full, query_spec) >= 2.5

    # Partial criteria match (category and subcategory, wrong color) -> 0.5
    cand_partial = {
        "id": 888,
        "master_category": "Footwear",
        "sub_category": "Shoes",
        "gender": "Men",
        "base_colour": "White",
        "price": 80.0,
    }
    assert evaluate_candidate_relevance(cand_partial, query_spec) == 0.5

    # Wrong gender demographic -> 0.0
    cand_wrong_gender = {
        "id": 777,
        "master_category": "Footwear",
        "sub_category": "Shoes",
        "gender": "Women",
        "base_colour": "Black",
    }
    assert evaluate_candidate_relevance(cand_wrong_gender, query_spec) == 0.0

    # Completely wrong category -> 0.0
    cand_wrong = {
        "id": 666,
        "master_category": "Apparel",
        "sub_category": "Topwear",
    }
    assert evaluate_candidate_relevance(cand_wrong, query_spec) == 0.0


def test_run_evaluation_mock(tmp_path):
    """Verifies that run_evaluation runs cleanly with mock queries and runner."""
    queries_file = tmp_path / "test_queries.json"
    output_file = tmp_path / "test_results.json"

    test_queries = [
        {
            "query_id": "mock_01",
            "query_type": "text",
            "query_text": "black running shoes",
            "relevant_ids": [101],
            "target_criteria": {"master_category": "Footwear", "sub_category": "Shoes"},
        }
    ]
    queries_file.write_text(json.dumps(test_queries), encoding="utf-8")

    with patch("src.evaluation.evaluator.BenchmarkRunner") as MockRunnerClass:
        mock_instance = MagicMock()
        MockRunnerClass.return_value = mock_instance

        # Naive returns id 999 (suboptimal)
        mock_instance.run_naive_knn.return_value = (
            [{"id": 999, "master_category": "Footwear", "score": 0.5}],
            15.0,
        )
        # Agent returns id 101 (exact match target)
        mock_instance.run_agentic.return_value = (
            [{"id": 101, "master_category": "Footwear", "score": 0.9}],
            45.0,
        )

        results = run_evaluation(
            queries_path=str(queries_file),
            output_path=str(output_file),
            limit_queries=1,
            top_k=5,
        )

        assert results["total_queries_evaluated"] == 1
        assert "naive_knn" in results
        assert "multi_agent_rag" in results
        assert results["multi_agent_rag"]["ndcg_at_10"] >= results["naive_knn"]["ndcg_at_10"]
        assert output_file.exists()
