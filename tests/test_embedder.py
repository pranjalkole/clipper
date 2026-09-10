import math

import pytest
import torch
from PIL import Image

from src.models.embedder import MultimodalEmbedder, get_embedder, reset_embedder


@pytest.fixture(scope="module")
def embedder():
    """Module-scoped embedder fixture to avoid reloading weights across tests."""
    return get_embedder()


def test_embedder_initialization(embedder):
    assert embedder is not None
    assert embedder.dim == 512
    assert isinstance(embedder.device, torch.device)
    assert embedder.model is not None


def test_alias_tag_resolution():
    """Verify alias mapping correctly handles hf-hub:apple/MobileCLIP2-S0."""
    alias_embedder = MultimodalEmbedder(
        model_name="MobileCLIP2-S0",
        pretrained_tag="hf-hub:apple/MobileCLIP2-S0",
    )
    assert alias_embedder.pretrained_tag == "dfndr2b"
    assert alias_embedder.dim == 512


def test_embed_text_shape_and_norm(embedder):
    text = "minimalist black running sneakers"
    vec = embedder.embed_text(text)

    assert isinstance(vec, list)
    assert len(vec) == 512
    # Verify unit norm: ||v|| = 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    assert norm == pytest.approx(1.0, rel=1e-4)


def test_embed_texts_batch(embedder):
    texts = [
        "casual cotton t-shirt",
        "leather formal shoes",
        "waterproof winter jacket",
    ]
    vectors = embedder.embed_texts(texts, batch_size=2)

    assert len(vectors) == 3
    for v in vectors:
        assert len(v) == 512
        norm = math.sqrt(sum(x * x for x in v))
        assert norm == pytest.approx(1.0, rel=1e-4)


def test_embed_text_empty_and_invalid(embedder):
    with pytest.raises(ValueError):
        embedder.embed_text("")

    with pytest.raises(ValueError):
        embedder.embed_text("   \n\t  ")

    with pytest.raises(ValueError):
        embedder.embed_text(None)  # type: ignore

    with pytest.raises(ValueError):
        embedder.embed_texts(["valid text", ""])


def test_embed_image_shape_and_norm(embedder):
    # Test RGB image of non-standard aspect ratio
    img_rgb = Image.new("RGB", (320, 240), color="blue")
    vec = embedder.embed_image(img_rgb)

    assert isinstance(vec, list)
    assert len(vec) == 512
    norm = math.sqrt(sum(x * x for x in vec))
    assert norm == pytest.approx(1.0, rel=1e-4)


def test_embed_image_formats(embedder):
    # Test RGBA and Grayscale (L) conversions
    img_rgba = Image.new("RGBA", (128, 128), color=(255, 0, 0, 128))
    vec_rgba = embedder.embed_image(img_rgba)
    assert len(vec_rgba) == 512
    norm_rgba = math.sqrt(sum(x * x for x in vec_rgba))
    assert norm_rgba == pytest.approx(1.0, rel=1e-4)

    img_gray = Image.new("L", (128, 128), color=128)
    vec_gray = embedder.embed_image(img_gray)
    assert len(vec_gray) == 512
    norm_gray = math.sqrt(sum(x * x for x in vec_gray))
    assert norm_gray == pytest.approx(1.0, rel=1e-4)


def test_embed_images_batch(embedder):
    images = [
        Image.new("RGB", (64, 64), color="red"),
        Image.new("RGB", (64, 64), color="green"),
        Image.new("RGB", (64, 64), color="blue"),
    ]
    vectors = embedder.embed_images(images, batch_size=2)

    assert len(vectors) == 3
    for v in vectors:
        assert len(v) == 512
        norm = math.sqrt(sum(x * x for x in v))
        assert norm == pytest.approx(1.0, rel=1e-4)


def test_embed_image_invalid(embedder):
    with pytest.raises(TypeError):
        embedder.embed_image("not an image")  # type: ignore

    with pytest.raises(TypeError):
        embedder.embed_images([Image.new("RGB", (64, 64)), "invalid"])  # type: ignore


def test_semantic_similarity(embedder):
    # Create distinct color images
    red_img = Image.new("RGB", (128, 128), color=(255, 0, 0))
    blue_img = Image.new("RGB", (128, 128), color=(0, 0, 255))

    v_red_img = embedder.embed_image(red_img)
    v_blue_img = embedder.embed_image(blue_img)

    v_red_text = embedder.embed_text("a vibrant red color")
    v_blue_text = embedder.embed_text("a deep blue color")

    sim_red_match = embedder.compute_similarity(v_red_text, v_red_img)
    sim_red_cross = embedder.compute_similarity(v_red_text, v_blue_img)

    sim_blue_match = embedder.compute_similarity(v_blue_text, v_blue_img)
    sim_blue_cross = embedder.compute_similarity(v_blue_text, v_red_img)

    assert sim_red_match > sim_red_cross
    assert sim_blue_match > sim_blue_cross


def test_singleton_get_embedder():
    inst1 = get_embedder()
    inst2 = get_embedder()
    assert inst1 is inst2

    reset_embedder()
    inst3 = get_embedder()
    assert inst3 is not None
