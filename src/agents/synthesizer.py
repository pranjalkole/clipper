"""
Multimodal synthesis node using Google Gemini 1.5 Flash.
Generates grounded e-commerce recommendations and visual citations.
Includes deterministic fallback for offline and CI testing environments.
"""

import logging
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel

from src.agents.state import SearchAgentState

logger = logging.getLogger(__name__)


def _deterministic_synthesis(
    query: str,
    candidates: list[dict[str, Any]],
    relaxed_filters: list[str],
    confidence_score: float,
) -> str:
    """
    Fallback synthesis producing clean, grounded Markdown recommendations.
    """
    if not candidates:
        msg = f"No matching products were found for '{query}'."
        if relaxed_filters:
            msg += f" (Note: We attempted to relax constraints: {', '.join(relaxed_filters)})."
        return msg

    lines = [
        f"### Curated Recommendations for '{query}'\n",
        f"Found **{len(candidates)}** matching items (Confidence: {confidence_score:.2f}):\n",
    ]

    for idx, c in enumerate(candidates[:5], start=1):
        name = c.get("product_display_name", f"Product #{c.get('id', idx)}")
        brand = c.get("brand", "Generic")
        price = c.get("price", 0.0)
        rating = c.get("rating", 4.0)
        score = c.get("score", 0.0)
        category = c.get("sub_category", c.get("master_category", "Apparel"))
        lines.append(
            f"{idx}. **{name}** by *{brand}* — **${price:.2f}** | Rating: {rating:.1f}★ | Match: {score:.2f}\n"
            f"   - Category: {category} | In Stock: {'Yes' if c.get('in_stock', True) else 'No'}"
        )

    if relaxed_filters:
        lines.append("\n> [!NOTE]")
        lines.append(
            f"> During search optimization, the following adjustments were made: {'; '.join(relaxed_filters)}."
        )

    return "\n".join(lines)


def create_synthesizer_node(
    llm: BaseChatModel | None = None,
) -> Callable[[SearchAgentState], dict[str, Any]]:
    """
    Factory creating the synthesizer node with Gemini 1.5 Flash or deterministic fallback.
    """

    def synthesizer_node(state: SearchAgentState) -> dict[str, Any]:
        query = state.get("query_text") or state.get("extracted_keywords") or "your search"
        candidates = state.get("candidates", [])
        relaxed_filters = state.get("relaxed_filters", [])
        confidence_score = state.get("confidence_score", 0.0)

        # Attempt LLM synthesis if model is configured
        if llm is not None and candidates:
            try:
                candidate_summaries = []
                for idx, c in enumerate(candidates[:5], start=1):
                    candidate_summaries.append(
                        f"Item {idx}: [ID {c.get('id')}] {c.get('product_display_name')}, Brand: {c.get('brand')}, "
                        f"Price: ${c.get('price', 0.0)}, Rating: {c.get('rating', 4.0)} stars, "
                        f"Category: {c.get('master_category')}/{c.get('sub_category')}, Score: {c.get('score', 0.0)}"
                    )
                context_str = "\n".join(candidate_summaries)
                relax_note = f"Adjustments made: {', '.join(relaxed_filters)}" if relaxed_filters else "None"

                prompt = (
                    "You are a personal fashion and catalog stylist AI assistant. "
                    "Synthesize a helpful, conversational, and grounded recommendation response for the user's query.\n\n"
                    f"User Query: '{query}'\n"
                    f"Filter Adjustments: {relax_note}\n\n"
                    f"Retrieved Candidates:\n{context_str}\n\n"
                    "Instructions:\n"
                    "1. Recommend the best matching products, citing their exact name, brand, price, and rating.\n"
                    "2. Explain why they fit the user's aesthetic and constraints.\n"
                    "3. If any filters were adjusted/relaxed, transparently mention it.\n"
                    "4. Do NOT invent products not listed in the retrieved candidates."
                )
                response = llm.invoke(prompt)
                final_text = response.content if hasattr(response, "content") else str(response)
                return {"final_response": str(final_text)}
            except Exception as exc:
                logger.warning("LLM synthesis failed, using fallback synthesizer: %s", exc)

        fallback_text = _deterministic_synthesis(
            query=query,
            candidates=candidates,
            relaxed_filters=relaxed_filters,
            confidence_score=confidence_score,
        )
        return {"final_response": fallback_text}

    return synthesizer_node
