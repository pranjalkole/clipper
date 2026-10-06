"""
Self-reflection & candidate critique node.
Evaluates candidate similarity scores and metadata match rates.
Executes dynamic query reformulation and filter relaxation loops when score < 0.70.
"""

import logging
from typing import Any, Callable

from src.agents.state import SearchAgentState
from src.core.config import settings

logger = logging.getLogger(__name__)


def _compute_reflection_metrics(
    candidates: list[dict[str, Any]],
    current_filters: dict[str, Any],
) -> tuple[float, float, float]:
    """
    Computes (semantic_score, metadata_match_rate, composite_reflection_score).
    """
    if not candidates:
        return 0.0, 0.0, 0.0

    scores = [float(c.get("score", 0.0)) for c in candidates]
    top_score = max(scores) if scores else 0.0
    avg_score = sum(scores) / len(scores) if scores else 0.0

    # Semantic component: blend top score (70%) and pool average (30%)
    semantic_score = 0.7 * top_score + 0.3 * avg_score

    # Metadata constraint satisfaction check
    if not current_filters:
        metadata_match_rate = 1.0
    else:
        match_count = 0
        for cand in candidates:
            cand_matches = True
            if "max_price" in current_filters and cand.get("price", 0.0) > current_filters["max_price"]:
                cand_matches = False
            if "min_price" in current_filters and cand.get("price", 0.0) < current_filters["min_price"]:
                cand_matches = False
            if "min_rating" in current_filters and cand.get("rating", 0.0) < current_filters["min_rating"]:
                cand_matches = False
            if "brand" in current_filters and str(current_filters["brand"]).lower() != str(cand.get("brand", "")).lower():
                cand_matches = False
            if "gender" in current_filters and str(current_filters["gender"]).lower() != str(cand.get("gender", "")).lower():
                cand_matches = False
            if cand_matches:
                match_count += 1
        metadata_match_rate = match_count / len(candidates)

    composite_score = 0.7 * semantic_score + 0.3 * metadata_match_rate
    return round(semantic_score, 4), round(metadata_match_rate, 4), round(composite_score, 4)


def create_reflection_node(
    min_confidence_score: float | None = None,
    max_retries: int | None = None,
) -> Callable[[SearchAgentState], dict[str, Any]]:
    """
    Factory creating the Self-Reflection & Critique node.
    """
    target_threshold = (
        min_confidence_score if min_confidence_score is not None else settings.min_confidence_score
    )
    target_max_retries = (
        max_retries if max_retries is not None else settings.max_reflection_retries
    )

    def reflection_node(state: SearchAgentState) -> dict[str, Any]:
        candidates = state.get("candidates", [])
        current_filters = dict(state.get("current_filters") or {})
        relaxed_filters = list(state.get("relaxed_filters") or [])
        retry_count = state.get("retry_count", 0)
        extracted_keywords = state.get("extracted_keywords") or state.get("query_text") or ""

        semantic_score, meta_score, reflection_score = _compute_reflection_metrics(
            candidates, current_filters
        )

        logger.info(
            "Self-Reflection assessment: score=%.4f (semantic=%.4f, meta=%.4f), threshold=%.2f, retry=%d/%d",
            reflection_score,
            semantic_score,
            meta_score,
            target_threshold,
            retry_count,
            target_max_retries,
        )

        # Check termination condition:
        # Either confidence is acceptable, OR we have exhausted max retries
        if reflection_score >= target_threshold or retry_count >= target_max_retries:
            if reflection_score >= target_threshold:
                note = f"Candidate pool verified successfully (confidence={reflection_score:.2f} >= {target_threshold:.2f})."
            else:
                note = (
                    f"Max reflection retries ({target_max_retries}) reached with best confidence "
                    f"{reflection_score:.2f}. Proceeding to synthesis."
                )

            return {
                "reflection_score": reflection_score,
                "confidence_score": reflection_score,
                "route": "proceed",
                "reflection_notes": [note],
            }

        # Sub-optimal candidates: trigger query reformulation and filter relaxation
        new_retry_count = retry_count + 1
        relaxed_note = ""

        # Priority 1: Expand max_price by 50%
        if "max_price" in current_filters:
            old_price = current_filters["max_price"]
            new_price = round(old_price * 1.5, 2)
            current_filters["max_price"] = new_price
            relaxed_filters.append(f"max_price expanded from ${old_price} to ${new_price}")
            relaxed_note = f"[Reflection Loop {new_retry_count}] Relaxed budget from ${old_price} to ${new_price} due to score {reflection_score:.2f}."

        # Priority 2: Drop restrictive brand constraint
        elif "brand" in current_filters:
            dropped_brand = current_filters.pop("brand")
            relaxed_filters.append(f"dropped brand constraint '{dropped_brand}'")
            relaxed_note = f"[Reflection Loop {new_retry_count}] Dropped brand constraint '{dropped_brand}' to broaden selection."

        # Priority 3: Drop sub_category, preserving master_category
        elif "sub_category" in current_filters:
            dropped_sub = current_filters.pop("sub_category")
            relaxed_filters.append(f"broadened sub_category '{dropped_sub}'")
            relaxed_note = f"[Reflection Loop {new_retry_count}] Broadened sub_category constraint from '{dropped_sub}'."

        # Priority 4: Relax in_stock_only
        elif current_filters.get("in_stock_only"):
            current_filters.pop("in_stock_only", None)
            relaxed_filters.append("relaxed in_stock_only requirement")
            relaxed_note = f"[Reflection Loop {new_retry_count}] Relaxed in_stock constraint."

        # Priority 5: Reformulate query keywords
        else:
            tokens = extracted_keywords.split()
            if len(tokens) > 2:
                # Strip outermost token to broaden semantic retrieval
                extracted_keywords = " ".join(tokens[:-1])
                relaxed_filters.append(f"reformulated query keywords to '{extracted_keywords}'")
                relaxed_note = f"[Reflection Loop {new_retry_count}] Reformulated query to '{extracted_keywords}'."
            else:
                relaxed_note = f"[Reflection Loop {new_retry_count}] Re-querying with relaxed parameters."

        logger.info(
            "Self-Reflection triggered retry (%d/%d): %s",
            new_retry_count,
            target_max_retries,
            relaxed_note,
        )

        return {
            "current_filters": current_filters,
            "relaxed_filters": relaxed_filters,
            "extracted_keywords": extracted_keywords,
            "reflection_score": reflection_score,
            "confidence_score": reflection_score,
            "retry_count": new_retry_count,
            "route": "retry",
            "reflection_notes": [relaxed_note],
        }

    return reflection_node
