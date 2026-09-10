import logging
import threading
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

from src.core.config import settings
from src.core.device import get_device

logger = logging.getLogger(__name__)

# Known tag aliases for MobileCLIP2 models in OpenCLIP
TAG_ALIASES = {
    "hf-hub:apple/MobileCLIP2-S0": "dfndr2b",
    "apple/MobileCLIP2-S0": "dfndr2b",
    "apple/MobileCLIP2-S2": "dfndr2b",
    "apple/MobileCLIP2-B": "dfndr2b",
}


class MultimodalEmbedder:
    """
    Multimodal feature extractor powered by Apple's MobileCLIP2-S0 architecture.
    Extracts L2-normalized 512-dimensional vector representations for text queries and images.
    """

    def __init__(
        self,
        model_name: str | None = None,
        pretrained_tag: str | None = None,
        device: str | torch.device | None = None,
        checkpoint_dir: str | Path | None = None,
    ):
        self.model_name = model_name or settings.model_name
        raw_tag = pretrained_tag or settings.pretrained_tag

        # Map known aliases to valid OpenCLIP pretrained tags
        self.pretrained_tag = TAG_ALIASES.get(raw_tag, raw_tag)

        # Device allocation
        if device is None:
            self.device = get_device(settings.device)
        elif isinstance(device, str):
            self.device = get_device(device)
        else:
            self.device = device

        self.checkpoint_dir = Path(checkpoint_dir or settings.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Initializing MultimodalEmbedder: model=%s, tag=%s, device=%s",
            self.model_name,
            self.pretrained_tag,
            self.device,
        )

        # Load model and transforms via OpenCLIP
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name=self.model_name,
                pretrained=self.pretrained_tag,
                cache_dir=str(self.checkpoint_dir),
            )
        except Exception as exc:
            logger.error(
                "Failed to load model %s (%s): %s", self.model_name, self.pretrained_tag, exc
            )
            raise RuntimeError(
                f"Could not load OpenCLIP model '{self.model_name}' with tag '{self.pretrained_tag}'."
            ) from exc

        self.model = self.model.to(self.device)
        self.model.eval()

        # Load tokenizer
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

        # Embedding dimension (512 for MobileCLIP2-S0)
        self._dim = getattr(self.model, "visual", None)
        if hasattr(self.model, "text_projection") and self.model.text_projection is not None:
            if isinstance(self.model.text_projection, torch.Tensor):
                self._dim = self.model.text_projection.shape[1]
            else:
                self._dim = self.model.text_projection.out_features
        elif hasattr(self.model, "visual") and hasattr(self.model.visual, "output_dim"):
            self._dim = self.model.visual.output_dim
        else:
            self._dim = settings.embedding_dim

        logger.info("MultimodalEmbedder initialized successfully (dim=%d)", self._dim)

    @property
    def dim(self) -> int:
        """Vector dimensionality of the embedding space."""
        return self._dim

    @torch.inference_mode()
    def embed_text(self, text: str) -> list[float]:
        """
        Computes L2-normalized embedding vector for a single text string.

        Args:
            text: Query or description string.

        Returns:
            512-dimensional unit-norm float vector as a list.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        tokens = self.tokenizer([text.strip()]).to(self.device)
        text_features = self.model.encode_text(tokens)
        normalized_features = F.normalize(text_features, p=2, dim=-1)
        return normalized_features.squeeze(0).cpu().tolist()

    @torch.inference_mode()
    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Computes L2-normalized embeddings for a batch of text strings.

        Args:
            texts: List of strings to encode.
            batch_size: Batch size for inference chunks.

        Returns:
            List of 512-dimensional unit-norm float vectors.
        """
        if not texts:
            return []

        for idx, t in enumerate(texts):
            if not isinstance(t, str) or not t.strip():
                raise ValueError(f"Text at index {idx} must be a non-empty string.")

        embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            chunk = [t.strip() for t in texts[i : i + batch_size]]
            tokens = self.tokenizer(chunk).to(self.device)
            text_features = self.model.encode_text(tokens)
            normalized = F.normalize(text_features, p=2, dim=-1)
            embeddings.extend(normalized.cpu().tolist())

        return embeddings

    @torch.inference_mode()
    def embed_image(self, image: Image.Image) -> list[float]:
        """
        Computes L2-normalized embedding vector for a single PIL image.

        Args:
            image: PIL.Image instance.

        Returns:
            512-dimensional unit-norm float vector as a list.
        """
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL.Image.Image, got {type(image).__name__}")

        # Ensure image is in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(tensor)
        normalized = F.normalize(image_features, p=2, dim=-1)
        return normalized.squeeze(0).cpu().tolist()

    @torch.inference_mode()
    def embed_images(self, images: list[Image.Image], batch_size: int = 32) -> list[list[float]]:
        """
        Computes L2-normalized embeddings for a batch of PIL images.

        Args:
            images: List of PIL.Image instances.
            batch_size: Batch size for inference chunks.

        Returns:
            List of 512-dimensional unit-norm float vectors.
        """
        if not images:
            return []

        for idx, img in enumerate(images):
            if not isinstance(img, Image.Image):
                raise TypeError(
                    f"Item at index {idx} is {type(img).__name__}, expected PIL.Image.Image"
                )

        embeddings: list[list[float]] = []
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            tensors = [
                self.preprocess(img.convert("RGB") if img.mode != "RGB" else img) for img in chunk
            ]
            batch_tensor = torch.stack(tensors).to(self.device)
            image_features = self.model.encode_image(batch_tensor)
            normalized = F.normalize(image_features, p=2, dim=-1)
            embeddings.extend(normalized.cpu().tolist())

        return embeddings

    @staticmethod
    def compute_similarity(vector_a: list[float], vector_b: list[float]) -> float:
        """
        Computes cosine similarity between two unit-normalized vectors.
        Since vectors are L2-normalized, cosine similarity equals dot product.
        """
        if len(vector_a) != len(vector_b):
            raise ValueError(f"Vector dimension mismatch: {len(vector_a)} vs {len(vector_b)}")
        t_a = torch.tensor(vector_a, dtype=torch.float32)
        t_b = torch.tensor(vector_b, dtype=torch.float32)
        return float(torch.dot(t_a, t_b).item())


# Thread-safe Singleton Management
_embedder_instance: MultimodalEmbedder | None = None
_embedder_lock = threading.Lock()


def get_embedder() -> MultimodalEmbedder:
    """Returns the process-wide cached MultimodalEmbedder instance."""
    global _embedder_instance
    if _embedder_instance is None:
        with _embedder_lock:
            if _embedder_instance is None:
                _embedder_instance = MultimodalEmbedder()
    return _embedder_instance


def reset_embedder() -> None:
    """Resets the singleton embedder instance (primarily for testing)."""
    global _embedder_instance
    with _embedder_lock:
        _embedder_instance = None
