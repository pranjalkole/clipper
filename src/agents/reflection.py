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


def _matches_filter(req_val: Any, cand_val: Any, allow_substring: bool = False) -> bool:
    """Helper to check if candidate metadata matches requested filter (string or list of strings)."""
    if req_val is None:
        return True
    if cand_val is None:
        return False
    cand_str = str(cand_val).lower().strip()
    if isinstance(req_val, list):
        req_items = [str(item).lower().strip() for item in req_val if str(item).strip()]
        if allow_substring:
            return any(item == cand_str or item in cand_str for item in req_items)
        return any(item == cand_str for item in req_items)
    req_str = str(req_val).lower().strip()
    if allow_substring:
        return req_str == cand_str or req_str in cand_str
    return req_str == cand_str


def _format_filter_val(val: Any) -> str:
    """Formats filter value nicely for reflection notes."""
    if isinstance(val, list):
        return ", ".join(str(v) for v in val)
    return str(val)


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
            if "master_category" in current_filters and not _matches_filter(current_filters["master_category"], cand.get("master_category")):
                cand_matches = False
            if "sub_category" in current_filters and not _matches_filter(current_filters["sub_category"], cand.get("sub_category")):
                cand_matches = False
            if "brand" in current_filters and not _matches_filter(current_filters["brand"], cand.get("brand")):
                cand_matches = False
            if "gender" in current_filters and not _matches_filter(current_filters["gender"], cand.get("gender")):
                cand_matches = False
            if "base_colour" in current_filters and not _matches_filter(current_filters["base_colour"], cand.get("base_colour"), allow_substring=True):
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

        # Zero-shot visual similarity check: critique candidate pool for color drift
        is_zero_shot_image = (
            bool(state.get("query_image"))
            and not bool(state.get("query_text"))
            and "base_colour" not in current_filters
        )

        if is_zero_shot_image and retry_count == 0 and candidates:
            cand1 = candidates[0]
            cand1_score = float(cand1.get("score", 0.0))
            cand1_color = cand1.get("base_colour")
            cand1_sub = cand1.get("sub_category")

            if cand1_score >= 0.75 and cand1_color and str(cand1_color).lower() != "unknown":
                top_pool = candidates[:min(5, len(candidates))]
                matching_count = sum(
                    1 for c in top_pool
                    if str(c.get("base_colour", "")).lower() == str(cand1_color).lower()
                )
                match_ratio = matching_count / len(top_pool)

                if match_ratio < 0.50:
                    drift_pct = int((1.0 - match_ratio) * 100)
                    new_filters = dict(current_filters)
                    new_filters["base_colour"] = cand1_color
                    if cand1_sub and str(cand1_sub).lower() not in ["general", "unknown", "item"]:
                        new_filters["sub_category"] = cand1_sub

                    critique_note = (
                        f"[Reflection Loop 1] Visual drift critique: primary visual match is '{cand1_color}' "
                        f"'{cand1_sub or 'item'}' ({cand1_score:.2f} confidence), but {drift_pct}% of "
                        f"top candidates drifted to other colors. Re-retrieving with active '{cand1_color}' constraint."
                    )
                    logger.info("Self-Reflection zero-shot visual critique: %s", critique_note)
                    return {
                        "current_filters": new_filters,
                        "relaxed_filters": [f"prioritized visual color '{cand1_color}'"],
                        "reflection_score": reflection_score,
                        "confidence_score": reflection_score,
                        "retry_count": 1,
                        "route": "retry",
                        "reflection_notes": [critique_note],
                    }

        # Check termination condition:
        # Either confidence is acceptable, OR we have exhausted max retries
        if reflection_score >= target_threshold or retry_count >= target_max_retries:
            if reflection_score >= target_threshold:
                if any("prioritized visual color" in f for f in relaxed_filters):
                    note = (
                        f"[Reflection Loop {retry_count + 1}] Candidate pool verified successfully with "
                        f"'{current_filters.get('base_colour')}' color fidelity "
                        f"(confidence={reflection_score:.2f} >= {target_threshold:.2f}). Proceeding to synthesis."
                    )
                else:
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
        if "max_price" in current_filters and not any("max_price expanded" in f for f in relaxed_filters):
            old_price = current_filters["max_price"]
            new_price = round(old_price * 1.5, 2)
            current_filters["max_price"] = new_price
            relaxed_filters.append(f"max_price expanded from ${old_price} to ${new_price}")
            relaxed_note = f"[Reflection Loop {new_retry_count}] Relaxed budget from ${old_price} to ${new_price} due to score {reflection_score:.2f}."

        # Priority 2: Drop restrictive color constraint
        elif "base_colour" in current_filters:
            dropped_color = current_filters.pop("base_colour")
            val_str = _format_filter_val(dropped_color)
            relaxed_filters.append(f"dropped color constraint '{val_str}'")
            relaxed_note = f"[Reflection Loop {new_retry_count}] Relaxed color constraint '{val_str}' to broaden candidate variety."

        # Priority 3: Drop restrictive brand constraint
        elif "brand" in current_filters:
            dropped_brand = current_filters.pop("brand")
            val_str = _format_filter_val(dropped_brand)
            relaxed_filters.append(f"dropped brand constraint '{val_str}'")
            relaxed_note = f"[Reflection Loop {new_retry_count}] Dropped brand constraint '{val_str}' to broaden selection."

        # Priority 4: Drop sub_category
        elif "sub_category" in current_filters:
            dropped_sub = current_filters.pop("sub_category")
            val_str = _format_filter_val(dropped_sub)
            relaxed_filters.append(f"broadened sub_category '{val_str}'")
            relaxed_note = f"[Reflection Loop {new_retry_count}] Broadened sub_category constraint from '{val_str}'."

        # Priority 5: Drop master_category
        elif "master_category" in current_filters:
            dropped_master = current_filters.pop("master_category")
            val_str = _format_filter_val(dropped_master)
            relaxed_filters.append(f"broadened master_category '{val_str}'")
            relaxed_note = f"[Reflection Loop {new_retry_count}] Broadened master_category constraint from '{val_str}'."

        # Priority 6: Relax in_stock_only
        elif current_filters.get("in_stock_only"):
            current_filters.pop("in_stock_only", None)
            relaxed_filters.append("relaxed in_stock_only requirement")
            relaxed_note = f"[Reflection Loop {new_retry_count}] Relaxed in_stock constraint."

        # Priority 7: Reformulate query keywords
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
