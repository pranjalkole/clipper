"""
Automated Evaluation Suite and Benchmark Harness.
Computes NDCG@10, Recall@5, Recall@10, MRR, latency distribution, and
comparative benchmarking between Multi-Agent Self-Reflective Search and Naive 1-turn k-NN.
"""

import argparse
import base64
import json
import logging
import math
from pathlib import Path
import time
from typing import Any

import httpx

from src.core.config import settings

logger = logging.getLogger("evaluator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# ============================================================================
# 1. Information Retrieval (IR) Metric Mathematical Formulations
# ============================================================================

def compute_dcg(relevance_scores: list[float], k: int = 10) -> float:
    """
    Computes Discounted Cumulative Gain at rank k:
      DCG@k = sum_{i=1}^k (2^{rel_i} - 1) / log2(i + 1)
    """
    scores = relevance_scores[:k]
    dcg = 0.0
    for idx, rel in enumerate(scores, start=1):
        if rel > 0:
            dcg += (2.0 ** float(rel) - 1.0) / math.log2(idx + 1.0)
    return dcg


def compute_ndcg(
    relevance_scores: list[float],
    ideal_relevance_scores: list[float],
    k: int = 10,
) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at rank k:
      NDCG@k = DCG@k / IDCG@k
    Returns 1.0 if both DCG and IDCG are 0.0.
    """
    idcg = compute_dcg(sorted(ideal_relevance_scores, reverse=True), k=k)
    if idcg <= 0.0:
        return 1.0 if not any(r > 0 for r in relevance_scores[:k]) else 0.0
    dcg = compute_dcg(relevance_scores, k=k)
    return min(1.0, max(0.0, dcg / idcg))


def compute_recall(
    retrieved_ids: list[int],
    relevant_ids: set[int] | list[int],
    k: int = 10,
) -> float:
    """
    Computes Recall at rank k:
      Recall@k = |Retrieved@k ∩ Relevant| / |Relevant|
    """
    rel_set = set(relevant_ids)
    if not rel_set:
        return 0.0
    retrieved_k = set(retrieved_ids[:k])
    return len(retrieved_k.intersection(rel_set)) / len(rel_set)


def compute_reciprocal_rank(
    retrieved_ids: list[int],
    relevant_ids: set[int] | list[int],
    k: int = 10,
) -> float:
    """
    Computes Reciprocal Rank (RR): 1 / rank of the first relevant candidate.
    Returns 0.0 if no relevant candidate is found in top k.
    """
    rel_set = set(relevant_ids)
    for idx, cand_id in enumerate(retrieved_ids[:k], start=1):
        if cand_id in rel_set:
            return 1.0 / float(idx)
    return 0.0


def compute_latency_distribution(latencies_ms: list[float]) -> dict[str, float]:
    """
    Computes P50, P90, P95, P99, mean, min, and max from a list of latencies.
    """
    if not latencies_ms:
        return {
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    s = sorted(latencies_ms)
    n = len(s)

    def percentile(p: float) -> float:
        idx = int(math.ceil((p / 100.0) * n)) - 1
        return round(s[max(0, min(idx, n - 1))], 2)

    return {
        "p50": percentile(50),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
        "mean": round(sum(s) / n, 2),
        "min": round(min(s), 2),
        "max": round(max(s), 2),
    }


# ============================================================================
# 2. Graded Relevance Judgment Evaluator
# ============================================================================

def evaluate_candidate_relevance(candidate: dict[str, Any], query_spec: dict[str, Any]) -> float:
    """
    Determines graded relevance score (0.0 to 3.0) for a candidate given query ground truth.
    - 3.0: Exact target ID match or complete satisfaction of all specified criteria
    - 2.0 - 2.5: Meets target category, gender, and primary attributes with minor variance
    - 0.5 - 1.0: Broad categorical match with significant constraint violations (color, price)
    - 0.0: Irrelevant, wrong demographic gender, or out-of-category match
    """
    cand_id = candidate.get("id")
    target_ids = query_spec.get("relevant_ids", [])
    if cand_id in target_ids:
        return 3.0

    target_criteria = query_spec.get("target_criteria", {})
    if not target_criteria:
        return 2.0

    target_master = target_criteria.get("master_category")
    target_sub = target_criteria.get("sub_category")
    cand_master = candidate.get("master_category")
    cand_sub = candidate.get("sub_category")

    # 1. Master category check (strict)
    if target_master and cand_master and target_master.lower() != cand_master.lower():
        return 0.0

    # 2. Gender demographic check (strict)
    target_gender = target_criteria.get("gender")
    if target_gender:
        cand_gender = str(candidate.get("gender", "")).lower()
        if cand_gender and cand_gender != "unisex" and target_gender.lower() != cand_gender:
            return 0.0

    # 3. Brand check (strict if requested)
    target_brand = target_criteria.get("brand")
    if target_brand:
        cand_brand = str(candidate.get("brand", "")).lower()
        if isinstance(target_brand, list):
            if not any(b.lower() == cand_brand for b in target_brand):
                return 0.2
        elif target_brand.lower() != cand_brand:
            return 0.2

    # 4. Color check
    target_color = target_criteria.get("base_colour")
    if target_color:
        cand_color = str(candidate.get("base_colour", "")).lower()
        if isinstance(target_color, list):
            if not any(str(c).lower() in cand_color for c in target_color):
                return 0.5
        elif str(target_color).lower() not in cand_color:
            return 0.5

    # 5. Price budget boundary
    target_max_price = target_criteria.get("max_price")
    if target_max_price is not None:
        cand_price = candidate.get("price", 0.0)
        if cand_price > target_max_price * 1.15:
            return 0.5

    # 6. Subcategory check
    if target_sub:
        if cand_sub and target_sub.lower() not in cand_sub.lower():
            return 1.5

    return 3.0


# ============================================================================
# 3. Benchmark Execution Runners (Direct & API Modes)
# ============================================================================

class BenchmarkRunner:
    """
    Executes benchmark queries across Naive 1-turn k-NN and Multi-Agent Self-Reflective search.
    Supports in-process direct execution and local HTTP API mode.
    """

    def __init__(self, mode: str = "auto", api_url: str = "http://127.0.0.1:8000"):
        self.api_url = api_url.rstrip("/")
        self.mode = mode
        self.http_client = httpx.Client(timeout=30.0)
        self._init_runtime()

    def _init_runtime(self) -> None:
        """Determines execution mode and initializes direct components if possible."""
        if self.mode == "api":
            logger.info("Using HTTP API execution mode at %s", self.api_url)
            self.use_api = True
            return

        if self.mode == "direct":
            logger.info("Using direct in-process execution mode")
            self.use_api = False
            self._load_direct_components()
            return

        # Auto-detection: attempt to connect to API or test direct access
        try:
            resp = self.http_client.get(f"{self.api_url}/health")
            if resp.status_code == 200 and resp.json().get("status") == "healthy":
                logger.info("Detected running API at %s. Using API mode.", self.api_url)
                self.use_api = True
                return
        except Exception:
            pass

        logger.info("API not reachable; defaulting to direct in-process execution.")
        self.use_api = False
        self._load_direct_components()

    def _load_direct_components(self) -> None:
        """Loads direct database and agent singletons."""
        from src.agents.graph import get_search_graph
        from src.db.qdrant import get_qdrant_client
        from src.models.embedder import get_embedder

        self.db_client = get_qdrant_client()
        self.embedder = get_embedder()
        self.search_graph = get_search_graph()

    def run_naive_knn(self, query: dict[str, Any], limit: int = 10) -> tuple[list[dict[str, Any]], float]:
        """
        Executes raw 1-turn nearest-neighbor vector search.
        """
        t0 = time.perf_counter()
        query_type = query.get("query_type", "text")

        if self.use_api:
            payload: dict[str, Any] = {"limit": limit}
            if query_type == "text":
                payload["query_text"] = query.get("query_text")
            elif query_type in ["visual", "composed"]:
                img_path = query.get("query_image_path")
                if img_path and Path(img_path).exists():
                    payload["query_image_path"] = img_path
                else:
                    payload["query_text"] = query.get("modification_text") or "fashion item"

            resp = self.http_client.post(f"{self.api_url}/search/knn", json=payload)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if resp.status_code == 200:
                candidates = resp.json().get("candidates", [])
                return candidates, elapsed_ms
            logger.warning("Naive k-NN API error %d: %s", resp.status_code, resp.text)
            return [], elapsed_ms

        # Direct in-process
        vector: list[float] = []
        if query_type == "text" and query.get("query_text"):
            vector = self.embedder.embed_text(query["query_text"])
        elif query.get("query_image_path") and Path(query["query_image_path"]).exists():
            from PIL import Image
            img = Image.open(query["query_image_path"]).convert("RGB")
            vector = self.embedder.embed_image(img)
        elif query.get("modification_text"):
            vector = self.embedder.embed_text(query["modification_text"])

        if not vector:
            return [], 0.0

        scored = self.db_client.search(query_vector=vector, limit=limit)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        candidates = [
            {**(p.payload or {}), "score": p.score, "rank": idx}
            for idx, p in enumerate(scored, start=1)
        ]
        return candidates, elapsed_ms

    def run_agentic(self, query: dict[str, Any], limit: int = 10) -> tuple[list[dict[str, Any]], float]:
        """
        Executes LangGraph multi-agent self-reflective search pipeline.
        """
        t0 = time.perf_counter()
        query_type = query.get("query_type", "text")

        if self.use_api:
            if query_type == "text":
                payload = {
                    "query_text": query.get("query_text"),
                    "filters": query.get("filters"),
                    "limit": limit,
                }
                resp = self.http_client.post(f"{self.api_url}/search/text", json=payload)
            elif query_type == "visual":
                img_path = query.get("query_image_path")
                if img_path and Path(img_path).exists():
                    with open(img_path, "rb") as f:
                        resp = self.http_client.post(
                            f"{self.api_url}/search/image",
                            files={"file": (Path(img_path).name, f, "image/jpeg")},
                            data={"limit": str(limit)},
                        )
                else:
                    resp = self.http_client.post(
                        f"{self.api_url}/search/text",
                        json={"query_text": "shoes", "limit": limit},
                    )
            elif query_type == "composed":
                img_path = query.get("query_image_path")
                mod_text = query.get("modification_text") or "similar item"
                if img_path and Path(img_path).exists():
                    with open(img_path, "rb") as f:
                        resp = self.http_client.post(
                            f"{self.api_url}/search/hybrid",
                            files={"file": (Path(img_path).name, f, "image/jpeg")},
                            data={"modification_text": mod_text, "limit": str(limit)},
                        )
                else:
                    resp = self.http_client.post(
                        f"{self.api_url}/search/text",
                        json={"query_text": mod_text, "limit": limit},
                    )
            else:
                resp = self.http_client.post(
                    f"{self.api_url}/search/text",
                    json={"query_text": query.get("query_text", ""), "limit": limit},
                )

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if resp.status_code == 200:
                data = resp.json()
                return data.get("candidates", []), elapsed_ms
            logger.warning("Agentic API error %d: %s", resp.status_code, resp.text)
            return [], elapsed_ms

        # Direct in-process execution via run_search_agent
        from src.agents.graph import run_search_agent
        img_spec = None
        if query.get("query_image_path") and Path(query["query_image_path"]).exists():
            img_bytes = Path(query["query_image_path"]).read_bytes()
            img_spec = f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode('utf-8')}"

        response = run_search_agent(
            query_text=query.get("query_text"),
            query_image=img_spec,
            modification_text=query.get("modification_text"),
            filters=query.get("filters"),
            limit=limit,
            graph=self.search_graph,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        candidates = [c.model_dump() for c in response.candidates]
        return candidates, elapsed_ms


# ============================================================================
# 4. Evaluation Suite Orchestrator
# ============================================================================

def run_evaluation(
    queries_path: str = "data/eval_queries.json",
    output_path: str = "data/eval_results.json",
    limit_queries: int | None = None,
    mode: str = "auto",
    api_url: str = "http://127.0.0.1:8000",
    top_k: int = 10,
) -> dict[str, Any]:
    """
    Runs the full evaluation benchmark across all queries, computing metrics
    for Naive k-NN and Multi-Agent Search, and exports the final results.
    """
    q_file = Path(queries_path)
    if not q_file.exists():
        raise FileNotFoundError(f"Queries file not found at {queries_path}")

    with open(q_file, "r", encoding="utf-8") as f:
        queries: list[dict[str, Any]] = json.load(f)

    if limit_queries is not None:
        queries = queries[:limit_queries]

    runner = BenchmarkRunner(mode=mode, api_url=api_url)

    naive_ndcgs: list[float] = []
    naive_recalls_5: list[float] = []
    naive_recalls_10: list[float] = []
    naive_mrrs: list[float] = []
    naive_latencies: list[float] = []

    agent_ndcgs: list[float] = []
    agent_recalls_5: list[float] = []
    agent_recalls_10: list[float] = []
    agent_mrrs: list[float] = []
    agent_latencies: list[float] = []

    logger.info("Starting evaluation across %d queries (top_k=%d)...", len(queries), top_k)

    for idx, query in enumerate(queries, start=1):
        q_id = query.get("query_id", f"query_{idx}")
        q_type = query.get("query_type", "text")
        relevant_target_ids = set(query.get("relevant_ids", []))

        # 1. Evaluate Naive 1-turn k-NN
        naive_cands, naive_lat = runner.run_naive_knn(query, limit=top_k)
        naive_latencies.append(naive_lat)
        naive_cand_ids = [c.get("id") for c in naive_cands]
        naive_rel_scores = [evaluate_candidate_relevance(c, query) for c in naive_cands]

        # 2. Evaluate Multi-Agent Self-Reflective Search
        agent_cands, agent_lat = runner.run_agentic(query, limit=top_k)
        agent_latencies.append(agent_lat)
        agent_cand_ids = [c.get("id") for c in agent_cands]
        agent_rel_scores = [evaluate_candidate_relevance(c, query) for c in agent_cands]

        # Ideal relevance: fully satisfying top_k candidates with score 3.0
        ideal_scores = [3.0] * top_k

        # Ground truth relevant set includes target IDs and candidates with score >= 2.0
        naive_relevant_set = {cid for cid, r in zip(naive_cand_ids, naive_rel_scores) if r >= 2.0 or cid in relevant_target_ids} | relevant_target_ids
        agent_relevant_set = {cid for cid, r in zip(agent_cand_ids, agent_rel_scores) if r >= 2.0 or cid in relevant_target_ids} | relevant_target_ids
        combined_relevant = naive_relevant_set | agent_relevant_set

        # Metrics for Naive
        naive_ndcg = compute_ndcg(naive_rel_scores, ideal_scores, k=top_k)
        naive_r5 = compute_recall(naive_cand_ids, combined_relevant, k=5)
        naive_r10 = compute_recall(naive_cand_ids, combined_relevant, k=10)
        naive_mrr = compute_reciprocal_rank(naive_cand_ids, combined_relevant, k=top_k)

        naive_ndcgs.append(naive_ndcg)
        naive_recalls_5.append(naive_r5)
        naive_recalls_10.append(naive_r10)
        naive_mrrs.append(naive_mrr)

        # Metrics for Agent
        agent_ndcg = compute_ndcg(agent_rel_scores, ideal_scores, k=top_k)
        agent_r5 = compute_recall(agent_cand_ids, combined_relevant, k=5)
        agent_r10 = compute_recall(agent_cand_ids, combined_relevant, k=10)
        agent_mrr = compute_reciprocal_rank(agent_cand_ids, combined_relevant, k=top_k)

        agent_ndcgs.append(agent_ndcg)
        agent_recalls_5.append(agent_r5)
        agent_recalls_10.append(agent_r10)
        agent_mrrs.append(agent_mrr)

        agent_ndcgs.append(agent_ndcg)
        agent_recalls_5.append(agent_r5)
        agent_recalls_10.append(agent_r10)
        agent_mrrs.append(agent_mrr)

        logger.info(
            "[%02d/%02d] %s (%s): Naive NDCG=%.3f (%.1fms) | Agent NDCG=%.3f (%.1fms)",
            idx,
            len(queries),
            q_id,
            q_type,
            naive_ndcg,
            naive_lat,
            agent_ndcg,
            agent_lat,
        )

    # Aggregate metrics
    mean_naive_ndcg = round(sum(naive_ndcgs) / len(naive_ndcgs), 4) if naive_ndcgs else 0.0
    mean_naive_r5 = round(sum(naive_recalls_5) / len(naive_recalls_5), 4) if naive_recalls_5 else 0.0
    mean_naive_r10 = round(sum(naive_recalls_10) / len(naive_recalls_10), 4) if naive_recalls_10 else 0.0
    mean_naive_mrr = round(sum(naive_mrrs) / len(naive_mrrs), 4) if naive_mrrs else 0.0
    naive_lat_dist = compute_latency_distribution(naive_latencies)

    mean_agent_ndcg = round(sum(agent_ndcgs) / len(agent_ndcgs), 4) if agent_ndcgs else 0.0
    mean_agent_r5 = round(sum(agent_recalls_5) / len(agent_recalls_5), 4) if agent_recalls_5 else 0.0
    mean_agent_r10 = round(sum(agent_recalls_10) / len(agent_recalls_10), 4) if agent_recalls_10 else 0.0
    mean_agent_mrr = round(sum(agent_mrrs) / len(agent_mrrs), 4) if agent_mrrs else 0.0
    agent_lat_dist = compute_latency_distribution(agent_latencies)

    # Comparative delta percentages
    def pct_delta(val: float, baseline: float) -> float:
        if baseline <= 0:
            return 0.0
        return round(((val - baseline) / baseline) * 100.0, 2)

    ndcg_delta = pct_delta(mean_agent_ndcg, mean_naive_ndcg)
    r5_delta = pct_delta(mean_agent_r5, mean_naive_r5)
    r10_delta = pct_delta(mean_agent_r10, mean_naive_r10)
    mrr_delta = pct_delta(mean_agent_mrr, mean_naive_mrr)

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_queries_evaluated": len(queries),
        "cutoff_k": top_k,
        "naive_knn": {
            "ndcg_at_10": mean_naive_ndcg,
            "recall_at_5": mean_naive_r5,
            "recall_at_10": mean_naive_r10,
            "mrr": mean_naive_mrr,
            "latency_ms": naive_lat_dist,
        },
        "multi_agent_rag": {
            "ndcg_at_10": mean_agent_ndcg,
            "recall_at_5": mean_agent_r5,
            "recall_at_10": mean_agent_r10,
            "mrr": mean_agent_mrr,
            "latency_ms": agent_lat_dist,
        },
        "comparative_delta": {
            "ndcg_at_10_improvement_pct": ndcg_delta,
            "recall_at_5_improvement_pct": r5_delta,
            "recall_at_10_improvement_pct": r10_delta,
            "mrr_improvement_pct": mrr_delta,
        },
    }

    # Save to JSON
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Evaluation results written to %s", out_file)

    # Print Formatted Summary Table
    print_summary_table(results)

    return results


def print_summary_table(results: dict[str, Any]) -> None:
    """Prints a clean, formatted ASCII table of the benchmark results."""
    naive = results["naive_knn"]
    agent = results["multi_agent_rag"]
    delta = results["comparative_delta"]

    print("\n" + "=" * 78)
    print("      MULTIMODAL SEARCH BENCHMARK & EVALUATION RESULTS SUMMARY")
    print("=" * 78)
    print(f"Total Queries Evaluated: {results['total_queries_evaluated']} | Metric Cutoff: k={results['cutoff_k']}")
    print("-" * 78)
    print(f"{'Metric':<25} | {'Naive 1-Turn k-NN':<18} | {'Multi-Agent RAG':<16} | {'Delta':<12}")
    print("-" * 78)
    print(f"{'NDCG@10':<25} | {naive['ndcg_at_10']:<18.4f} | {agent['ndcg_at_10']:<16.4f} | +{delta['ndcg_at_10_improvement_pct']}%")
    print(f"{'Recall@5':<25} | {naive['recall_at_5']:<18.4f} | {agent['recall_at_5']:<16.4f} | +{delta['recall_at_5_improvement_pct']}%")
    print(f"{'Recall@10':<25} | {naive['recall_at_10']:<18.4f} | {agent['recall_at_10']:<16.4f} | +{delta['recall_at_10_improvement_pct']}%")
    print(f"{'MRR':<25} | {naive['mrr']:<18.4f} | {agent['mrr']:<16.4f} | +{delta['mrr_improvement_pct']}%")
    print("-" * 78)
    print("LATENCY DISTRIBUTION (ms):")
    print(f"  P50 Latency            | {naive['latency_ms']['p50']:<18.2f} | {agent['latency_ms']['p50']:<16.2f} |")
    print(f"  P90 Latency            | {naive['latency_ms']['p90']:<18.2f} | {agent['latency_ms']['p90']:<16.2f} |")
    print(f"  P95 Latency            | {naive['latency_ms']['p95']:<18.2f} | {agent['latency_ms']['p95']:<16.2f} |")
    print(f"  P99 Latency            | {naive['latency_ms']['p99']:<18.2f} | {agent['latency_ms']['p99']:<16.2f} |")
    print(f"  Mean Latency           | {naive['latency_ms']['mean']:<18.2f} | {agent['latency_ms']['mean']:<16.2f} |")
    print("=" * 78 + "\n")


# ============================================================================
# 5. CLI Entrypoint
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated Evaluation Suite and Benchmark Harness for Multimodal Search"
    )
    parser.add_argument(
        "--queries",
        "-q",
        type=str,
        default="data/eval_queries.json",
        help="Path to evaluation queries JSON file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/eval_results.json",
        help="Path to export evaluation results JSON.",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Run on first N queries for rapid testing.",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["auto", "direct", "api"],
        default="auto",
        help="Execution mode (auto detects running API, direct uses local modules, api calls HTTP endpoint).",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Base URL of running FastAPI gateway.",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=10,
        help="Top-k candidates cutoff for NDCG and Recall metrics (default: 10).",
    )

    args = parser.parse_args()
    run_evaluation(
        queries_path=args.queries,
        output_path=args.output,
        limit_queries=args.limit,
        mode=args.mode,
        api_url=args.api_url,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
