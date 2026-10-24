"""
Router node for intent classification and structured constraint extraction.
Extracts scalar catalog filters and cleanses semantic search keywords.
"""

import json
import logging
import re
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel

from src.agents.state import SearchAgentState

logger = logging.getLogger(__name__)

# Known catalog categories and brands for heuristic extraction
KNOWN_MASTER_CATEGORIES = {
    "apparel": ["shirt", "t-shirt", "tshirt", "tee", "top", "dress", "pants", "jeans", "trousers", "jacket", "coat", "sweater", "hoodie", "shorts", "skirt", "suit", "blazer", "clothing", "clothes"],
    "footwear": ["shoe", "shoes", "sneaker", "sneakers", "boot", "boots", "sandal", "sandals", "heel", "heels", "loafer", "loafers", "flip flop", "footwear", "clogs"],
    "accessories": ["bag", "bags", "backpack", "wallet", "belt", "belts", "watch", "watches", "sunglasses", "glasses", "cap", "caps", "hat", "hats", "scarf", "tie", "socks", "jewellery", "jewelry"],
    "personal care": ["perfume", "fragrance", "cologne", "deodorant", "lotion", "lipstick", "makeup", "beauty"],
}

KNOWN_SUB_CATEGORIES = {
    "topwear": ["shirt", "t-shirt", "tshirt", "tee", "top", "sweater", "hoodie", "jacket"],
    "bottomwear": ["pants", "jeans", "trousers", "shorts", "skirt", "leggings"],
    "shoes": ["shoe", "shoes", "sneaker", "sneakers", "boot", "boots", "loafer", "loafers"],
    "bags": ["bag", "bags", "backpack", "handbag", "wallet"],
    "watches": ["watch", "watches", "timepiece"],
    "eyewear": ["sunglasses", "glasses", "shades"],
    "fragrance": ["perfume", "fragrance", "cologne", "deodorant"],
}

KNOWN_BRANDS = [
    "nike", "puma", "adidas", "reebok", "fila", "levi's", "levis", "united colors of benetton",
    "u.s. polo assn.", "us polo", "tommy hilfiger", "vans", "converse", "under armour",
    "zara", "h&m", "fastrack", "titan", "casio", "ray-ban", "oakley", "woodland", "wildcraft"
]

GENDER_MAP = {
    "men": "Men", "man": "Men", "male": "Men", "guy": "Men", "gentleman": "Men",
    "women": "Women", "woman": "Women", "female": "Women", "lady": "Women",
    "boys": "Boys", "boy": "Boys",
    "girls": "Girls", "girl": "Girls",
    "unisex": "Unisex",
}

KNOWN_COLORS: dict[str, str] = {
    "navy blue": "Navy Blue",
    "navy": "Navy Blue",
    "black": "Black",
    "white": "White",
    "blue": "Blue",
    "red": "Red",
    "grey": "Grey",
    "gray": "Grey",
    "green": "Green",
    "yellow": "Yellow",
    "pink": "Pink",
    "purple": "Purple",
    "brown": "Brown",
    "beige": "Beige",
    "orange": "Orange",
    "maroon": "Maroon",
    "olive": "Olive",
    "gold": "Gold",
    "silver": "Silver",
}


def _extract_heuristics(query_text: str) -> tuple[dict[str, Any], str]:
    """
    Rule-based extraction of filters and semantic query keywords from text.
    """
    text_lower = query_text.lower()
    filters: dict[str, Any] = {}
    tokens_to_remove: list[str] = []

    # 1. Price boundaries
    # e.g., "under $50", "below 100", "less than $45.50", "max price 80"
    under_match = re.search(r"(?:under|below|less\s+than|max(?:imum)?\s*(?:price)?)\s*\$?(\d+(?:\.\d+)?)", text_lower)
    if under_match:
        filters["max_price"] = float(under_match.group(1))
        tokens_to_remove.append(under_match.group(0))

    # e.g., "over $30", "above 40", "more than $25", "min price 20"
    over_match = re.search(r"(?:over|above|more\s+than|min(?:imum)?\s*(?:price)?)\s*\$?(\d+(?:\.\d+)?)", text_lower)
    if over_match:
        filters["min_price"] = float(over_match.group(1))
        tokens_to_remove.append(over_match.group(0))

    # e.g., "between $20 and $50", "$20 - $50"
    between_match = re.search(r"(?:between\s*)?\$?(\d+(?:\.\d+)?)\s*(?:and|-|to)\s*\$?(\d+(?:\.\d+)?)", text_lower)
    if between_match and "min_price" not in filters and "max_price" not in filters:
        filters["min_price"] = float(between_match.group(1))
        filters["max_price"] = float(between_match.group(2))
        tokens_to_remove.append(between_match.group(0))

    # 2. Rating boundaries
    rating_match = re.search(r"(?:rating\s*(?:>=|above|over|at\s+least)?\s*|rated\s+)?([1-5](?:\.\d+)?)\s*(?:\+|-\s*star|stars?|\*)", text_lower)
    if rating_match:
        filters["min_rating"] = float(rating_match.group(1))
        tokens_to_remove.append(rating_match.group(0))

    # 3. Stock availability
    if re.search(r"\b(in\s+stock|available)\b", text_lower):
        filters["in_stock_only"] = True
        tokens_to_remove.append(re.search(r"\b(in\s+stock|available)\b", text_lower).group(0))

    # 4. Gender
    for word, target_gender in GENDER_MAP.items():
        pattern = rf"\b(?:for\s+)?{word}\b"
        if re.search(pattern, text_lower):
            filters["gender"] = target_gender
            tokens_to_remove.append(re.search(pattern, text_lower).group(0))
            break

    # 5. Brand
    for brand in KNOWN_BRANDS:
        pattern = rf"\b{re.escape(brand)}\b"
        if re.search(pattern, text_lower):
            filters["brand"] = brand.title() if brand not in ["nike", "puma", "adidas"] else brand.capitalize()
            # Retain brand in semantic tokens for embedding, or leave as filter
            break

    # 6. Color
    for color_phrase, target_color in KNOWN_COLORS.items():
        pattern = rf"\b{re.escape(color_phrase)}\b"
        if re.search(pattern, text_lower):
            filters["base_colour"] = target_color
            break

    # 7. Master & Sub category
    for master, keywords in KNOWN_MASTER_CATEGORIES.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", text_lower):
                filters["master_category"] = master.capitalize()
                break
        if "master_category" in filters:
            break

    for sub, keywords in KNOWN_SUB_CATEGORIES.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", text_lower):
                filters["sub_category"] = sub.capitalize()
                break
        if "sub_category" in filters:
            break

    # Cleanse keywords for MobileCLIP2 embedding
    cleaned = text_lower
    # Remove phrases like "looking for", "find me", "show me", "i want"
    cleaned = re.sub(r"^(?:please\s+)?(?:find|show|give|search|look\s+for|i\s+(?:want|need|am\s+looking\s+for))\s+(?:me\s+)?(?:a|an|some)?\s*", "", cleaned).strip()
    for tok in tokens_to_remove:
        cleaned = cleaned.replace(tok, " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return filters, cleaned or query_text


def create_router_node(llm: BaseChatModel | None = None) -> Callable[[SearchAgentState], dict[str, Any]]:
    """
    Factory creating a router node with optional LLM-assisted intent extraction
    and resilient heuristic fallback.
    """

    def router_node(state: SearchAgentState) -> dict[str, Any]:
        query_text = state.get("query_text") or ""
        query_image = state.get("query_image")
        modification_text = state.get("modification_text")
        existing_filters = dict(state.get("current_filters") or {})

        # Check for composed image retrieval intent
        is_cir = bool(
            modification_text
            or (query_image and query_text and any(w in query_text.lower() for w in ["same", "similar", "instead", "change", "like this"]))
        )

        extracted_filters: dict[str, Any] = {}
        extracted_keywords = query_text

        if query_text.strip():
            # If LLM is provided, attempt structured intent extraction
            llm_extracted = False
            if llm is not None:
                try:
                    prompt = (
                        "You are an e-commerce query parser. Analyze the user query and extract scalar catalog filters "
                        "and the core semantic product search keywords.\n"
                        "Return ONLY valid JSON matching this schema:\n"
                        "{\n"
                        '  "master_category": str or null,\n'
                        '  "sub_category": str or null,\n'
                        '  "gender": "Men"|"Women"|"Boys"|"Girls"|"Unisex" or null,\n'
                        '  "base_colour": str or null,\n'
                        '  "brand": str or null,\n'
                        '  "min_price": float or null,\n'
                        '  "max_price": float or null,\n'
                        '  "min_rating": float or null,\n'
                        '  "in_stock_only": bool or null,\n'
                        '  "keywords": str\n'
                        "}\n"
                        f"Query: {query_text}"
                    )
                    resp = llm.invoke(prompt)
                    content = resp.content if hasattr(resp, "content") else str(resp)
                    # Extract JSON block
                    json_match = re.search(r"\{.*\}", str(content), re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group(0))
                        extracted_keywords = parsed.get("keywords") or query_text
                        for k, v in parsed.items():
                            if k != "keywords" and v is not None:
                                extracted_filters[k] = v
                        llm_extracted = True
                except Exception as exc:
                    logger.debug("LLM query parsing failed, falling back to heuristics: %s", exc)

            if not llm_extracted:
                heuristic_filters, cleaned_keywords = _extract_heuristics(query_text)
                extracted_filters = heuristic_filters
                extracted_keywords = cleaned_keywords

        # Merge with existing preset filters (explicit preset filters take precedence)
        merged_filters = {**extracted_filters, **existing_filters}

        logger.info(
            "Router processed query. is_cir=%s, keywords='%s', filters=%s",
            is_cir,
            extracted_keywords,
            merged_filters,
        )

        return {
            "is_composed_query": is_cir,
            "extracted_keywords": extracted_keywords,
            "extracted_filters": extracted_filters,
            "current_filters": merged_filters,
            "relaxed_filters": [],
        }

    return router_node
