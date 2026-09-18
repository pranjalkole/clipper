import hashlib
import re
from typing import Any

from pydantic import BaseModel, Field

# Well-known multi-word and single-word brands prevalent in fashion datasets
KNOWN_BRANDS = [
    "United Colors of Benetton",
    "Flying Machine",
    "French Connection",
    "U.S. Polo Assn.",
    "Jack & Jones",
    "Tommy Hilfiger",
    "Calvin Klein",
    "Louis Philippe",
    "Allen Solly",
    "Peter England",
    "Wrangler",
    "Levi's",
    "Roadster",
    "HRX",
    "Puma",
    "Nike",
    "Adidas",
    "Reebok",
    "Fila",
    "Converse",
    "Vans",
    "Titan",
    "Fastrack",
    "Casio",
    "Fossil",
    "Ray-Ban",
    "Oakley",
    "Turtle",
    "Fabindia",
    "Biba",
    "W",
    "Aurelia",
    "Global Desi",
    "Van Heusen",
    "Arrow",
    "Park Avenue",
    "John Players",
    "Spykar",
    "Lee",
    "Mufti",
    "Wildcraft",
    "American Tourister",
    "Skybags",
]

# Price boundaries per master category (min_price, max_price)
CATEGORY_PRICE_RANGES: dict[str, tuple[float, float]] = {
    "Apparel": (19.99, 149.99),
    "Footwear": (34.99, 199.99),
    "Accessories": (9.99, 89.99),
    "Personal Care": (7.99, 59.99),
    "Sporting Goods": (24.99, 179.99),
}
DEFAULT_PRICE_RANGE: tuple[float, float] = (14.99, 99.99)


class EnrichedCatalogItem(BaseModel):
    """Normalized and deterministically enriched catalog item representation."""

    id: int
    product_display_name: str
    master_category: str
    sub_category: str
    article_type: str
    base_colour: str
    gender: str
    season: str
    year: int | None = None
    usage: str
    brand: str
    price: float = Field(ge=0.0)
    rating: float = Field(ge=1.0, le=5.0)
    rating_count: int = Field(ge=0)
    in_stock: bool
    tags: list[str] = Field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        """Convert item to dictionary suitable for Qdrant payload."""
        return self.model_dump()


def _deterministic_hash(item_id: int | str, salt: str = "") -> int:
    """Computes a platform-independent deterministic hash integer from item_id and salt."""
    key = f"{item_id}:{salt}".encode()
    return int(hashlib.md5(key).hexdigest(), 16)


def extract_brand(display_name: str, item_id: int) -> str:
    """
    Extracts brand from product display name, falling back to a deterministic choice
    from KNOWN_BRANDS if display name does not specify one clearly.
    """
    if not display_name or not display_name.strip():
        idx = _deterministic_hash(item_id, "brand_fallback") % len(KNOWN_BRANDS)
        return KNOWN_BRANDS[idx]

    name = display_name.strip()

    # Check known multi-word / special brands first
    for brand in KNOWN_BRANDS:
        if name.lower().startswith(brand.lower() + " ") or name.lower() == brand.lower():
            return brand

    # Fallback to the first capitalized token or word
    first_token = name.split()[0]
    cleaned = re.sub(r"[^\w\'-]", "", first_token)
    if cleaned and len(cleaned) > 1:
        return cleaned

    idx = _deterministic_hash(item_id, "brand_fallback") % len(KNOWN_BRANDS)
    return KNOWN_BRANDS[idx]


def compute_deterministic_price(item_id: int, master_category: str) -> float:
    """
    Computes a realistic, deterministic price based on product ID and category.
    """
    min_price, max_price = CATEGORY_PRICE_RANGES.get(master_category, DEFAULT_PRICE_RANGE)
    h = _deterministic_hash(item_id, "price") % 10000
    ratio = h / 10000.0
    price = min_price + ratio * (max_price - min_price)
    return round(price, 2)


def compute_deterministic_rating(item_id: int) -> tuple[float, int]:
    """
    Computes a realistic, deterministic rating (3.2 - 5.0) and review count (10 - 1500).
    """
    rating_hash = _deterministic_hash(item_id, "rating") % 181  # 0 to 180
    rating = round(3.2 + (rating_hash / 100.0), 1)

    count_hash = _deterministic_hash(item_id, "rating_count") % 1491
    rating_count = 10 + count_hash

    return rating, rating_count


def compute_deterministic_stock(item_id: int) -> bool:
    """
    Computes deterministic in-stock status (85% probability in stock).
    """
    stock_hash = _deterministic_hash(item_id, "in_stock") % 100
    return stock_hash < 85


def generate_search_tags(*values: Any) -> list[str]:
    """
    Generates cleaned, deduplicated search tags from metadata values.
    """
    tags: set[str] = set()
    for val in values:
        if val is None:
            continue
        # Split tokens on spaces, hyphens, and slashes
        tokens = re.findall(r"[a-zA-Z0-9]+", str(val).lower())
        for token in tokens:
            if len(token) > 1 and not token.isdigit():
                tags.add(token)
    return sorted(tags)


def enrich_catalog_item(raw_record: dict[str, Any]) -> EnrichedCatalogItem:
    """
    Converts and enriches a raw dataset record into a standardized EnrichedCatalogItem.

    Supports both original dataset naming (camelCase: masterCategory, subCategory,
    articleType, baseColour, productDisplayName) and snake_case representations.
    """
    item_id = int(raw_record["id"])

    display_name = str(
        raw_record.get("productDisplayName") or raw_record.get("product_display_name") or ""
    ).strip()
    master_category = str(
        raw_record.get("masterCategory") or raw_record.get("master_category") or "Apparel"
    ).strip()
    sub_category = str(
        raw_record.get("subCategory") or raw_record.get("sub_category") or "General"
    ).strip()
    article_type = str(
        raw_record.get("articleType") or raw_record.get("article_type") or "Item"
    ).strip()
    base_colour = str(
        raw_record.get("baseColour") or raw_record.get("base_colour") or "Unknown"
    ).strip()
    gender = str(raw_record.get("gender") or "Unisex").strip()
    season = str(raw_record.get("season") or "All").strip()
    usage = str(raw_record.get("usage") or "Casual").strip()

    raw_year = raw_record.get("year")
    year: int | None = None
    if raw_year is not None and str(raw_year).strip() and not str(raw_year).lower().startswith("nan"):
        try:
            year = int(float(raw_year))
        except (ValueError, TypeError):
            year = None

    # Deterministic enrichments
    brand = extract_brand(display_name, item_id)
    price = compute_deterministic_price(item_id, master_category)
    rating, rating_count = compute_deterministic_rating(item_id)
    in_stock = compute_deterministic_stock(item_id)
    tags = generate_search_tags(
        gender, master_category, sub_category, article_type, base_colour, season, usage, brand
    )

    return EnrichedCatalogItem(
        id=item_id,
        product_display_name=display_name,
        master_category=master_category,
        sub_category=sub_category,
        article_type=article_type,
        base_colour=base_colour,
        gender=gender,
        season=season,
        year=year,
        usage=usage,
        brand=brand,
        price=price,
        rating=rating,
        rating_count=rating_count,
        in_stock=in_stock,
        tags=tags,
    )
