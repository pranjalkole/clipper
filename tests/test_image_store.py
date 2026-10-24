"""
Test suite for CatalogImageStore: lazy initialization, dataset retrieval, and disk caching.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image
import pytest

from src.pipeline.image_store import CatalogImageStore, get_image_store


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary directory for image caching."""
    cache_dir = tmp_path / "catalog_images"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def test_image_store_lazy_initialization(temp_cache_dir: Path):
    """Verifies that dataset is not loaded until first image retrieval."""
    store = CatalogImageStore(cache_dir=temp_cache_dir)
    assert store._id_to_idx is None
    assert store._dataset is None


def test_image_store_disk_cache_hit(temp_cache_dir: Path):
    """Verifies that existing cached image on disk is returned immediately without dataset query."""
    store = CatalogImageStore(cache_dir=temp_cache_dir)
    test_file = temp_cache_dir / "12345.jpg"
    test_file.write_bytes(b"dummy-cached-jpeg-bytes")

    data = store.get_image_bytes(12345)
    assert data == b"dummy-cached-jpeg-bytes"
    # Ensure dataset was not loaded
    assert store._dataset is None


def test_image_store_dataset_fetch_and_cache(temp_cache_dir: Path):
    """Verifies image retrieval from dataset and subsequent disk caching."""
    store = CatalogImageStore(cache_dir=temp_cache_dir)

    # Create dummy PIL image
    dummy_img = Image.new("RGB", (30, 40), color="blue")
    mock_dataset = [
        {"id": 1001, "image": dummy_img},
    ]

    with patch("src.pipeline.image_store.load_dataset") as mock_load:
        mock_ds = MagicMock()
        mock_ds.__len__.return_value = 1
        mock_ds.select_columns.return_value = [{"id": 1001}]
        mock_ds.__getitem__.side_effect = lambda idx: mock_dataset[idx]
        mock_load.return_value = mock_ds

        data = store.get_image_bytes(1001)
        assert data is not None
        assert data.startswith(b"\xff\xd8")  # JPEG signature

        # Verify disk cache file was created
        cached_file = temp_cache_dir / "1001.jpg"
        assert cached_file.is_file()
        assert cached_file.read_bytes() == data

        # Second call should read from disk directly
        data_second = store.get_image_bytes(1001)
        assert data_second == data


def test_image_store_unknown_product_id(temp_cache_dir: Path):
    """Verifies that an unknown product ID returns None."""
    store = CatalogImageStore(cache_dir=temp_cache_dir)
    with patch("src.pipeline.image_store.load_dataset") as mock_load:
        mock_ds = MagicMock()
        mock_ds.select_columns.return_value = [{"id": 1001}]
        mock_load.return_value = mock_ds

        data = store.get_image_bytes(9999)
        assert data is None


def test_singleton_get_image_store():
    """Verifies that get_image_store returns a persistent singleton."""
    store1 = get_image_store()
    store2 = get_image_store()
    assert store1 is store2
