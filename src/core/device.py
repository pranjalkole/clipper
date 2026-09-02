import logging
import torch

logger = logging.getLogger(__name__)


def get_device(preference: str = "auto") -> torch.device:
    """
    Detects and returns the best available hardware accelerator.

    Args:
        preference: "auto", "cuda", "mps", or "cpu".

    Returns:
        torch.device instance representing target compute device.
    """
    preference = preference.lower().strip()

    if preference != "auto":
        return torch.device(preference)

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info("Using CUDA hardware acceleration: %s", device_name)
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Using Apple Silicon MPS hardware acceleration")
        return torch.device("mps")

    logger.info("Using CPU device fallback")
    return torch.device("cpu")
