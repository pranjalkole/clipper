"""
Thread-safe catalog image store that resolves product images from HuggingFace dataset
and provides local disk caching for sub-millisecond retrieval.
"""

import io
import logging
import os
from pathlib import Path
import threading
from typing import Any
from datasets import load_dataset
from PIL import Image

from src.core.config import settings

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("./data/catalog_images")


class CatalogImageStore:
    """
    Manages access to catalog product images with an in-memory index
    and persistent disk-based caching.
    """

    def __init__(
        self,
        dataset_name: str | None = None,
        cache_dir: Path | str | None = None,
    ):
        self.dataset_name = dataset_name or settings.dataset_name
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._dataset: Any = None
        self._id_to_idx: dict[int, int] | None = None
        self._lock = threading.Lock()

    def _ensure_initialized(self) -> None:
        """Lazily loads the dataset and builds the ID-to-index mapping."""
        if self._id_to_idx is not None:
            return

        with self._lock:
            if self._id_to_idx is not None:
                return

            logger.info("Initializing CatalogImageStore with dataset: %s", self.dataset_name)
            ds = load_dataset(self.dataset_name, split="train")
            id_col = ds.select_columns(["id"])
            id_map: dict[int, int] = {}
            for idx, item in enumerate(id_col):
                id_map[int(item["id"])] = idx

            self._dataset = ds
            self._id_to_idx = id_map
            logger.info("CatalogImageStore ready with %d indexed product images", len(self._id_to_idx))

    def get_image_bytes(self, product_id: int) -> bytes | None:
        """
        Retrieves JPEG image bytes for a product ID.
        Checks local disk cache first; falls back to Hugging Face dataset.
        """
        # 1. Check local disk cache
        cache_file = self.cache_dir / f"{product_id}.jpg"
        if cache_file.is_file():
            try:
                return cache_file.read_bytes()
            except Exception as exc:
                logger.warning("Failed reading cached image %s: %s", cache_file, exc)

        # 2. Lazy dataset load and lookup
        try:
            self._ensure_initialized()
            if self._id_to_idx is None or self._dataset is None:
                return None

            idx = self._id_to_idx.get(product_id)
            if idx is None:
                logger.debug("Product ID %d not found in image store", product_id)
                return None

            raw_img = self._dataset[idx]["image"]
            if not isinstance(raw_img, Image.Image):
                return None

            # Convert to RGB JPEG bytes
            img_rgb = raw_img.convert("RGB")
            buf = io.BytesIO()
            img_rgb.save(buf, format="JPEG", quality=85)
            jpeg_bytes = buf.getvalue()

            # Save to disk cache
            try:
                cache_file.write_bytes(jpeg_bytes)
            except Exception as exc:
                logger.warning("Failed caching image to %s: %s", cache_file, exc)

            return jpeg_bytes
        except Exception as exc:
            logger.error("Error retrieving image for product %d: %s", product_id, exc)
            return None


# Global singleton
_image_store: CatalogImageStore | None = None
_store_lock = threading.Lock()


def get_image_store() -> CatalogImageStore:
    """Returns the singleton CatalogImageStore instance."""
    global _image_store
    if _image_store is None:
        with _store_lock:
            if _image_store is None:
                _image_store = CatalogImageStore()
    return _image_store
