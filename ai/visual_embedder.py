"""Visual embedder using DINOv2.

DINOv2 provides rich visual representations useful for visual similarity
scoring and secondary candidate ranking. It complements SigLIP2's
cross-modal semantic alignment with a purely visual similarity signal.

DINOv2 is NOT used for text-to-image retrieval.
It contributes the visual_score component in the final candidate ranking.
"""

import logging
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from utils.embeddings import normalize_embedding

logger: logging.Logger = logging.getLogger(__name__)

# Smallest DINOv2 variant available through HuggingFace transformers.
# Larger variants (base, large, giant) provide richer representations
# at increased memory and compute cost.
_MODEL_ID: str = "facebook/dinov2-small"


class VisualEmbedder:
    """Produces normalized DINOv2 visual embeddings for images.

    The model is loaded once and remains in memory for the application lifetime.
    Embeddings are unit-normalized for cosine similarity comparison.
    """

    def __init__(self) -> None:
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading VisualEmbedder (DINOv2) on device: %s", self._device)

        start: float = time.perf_counter()
        self._processor: AutoImageProcessor = AutoImageProcessor.from_pretrained(
            _MODEL_ID
        )
        self._model: AutoModel = AutoModel.from_pretrained(_MODEL_ID).to(self._device)
        self._model.eval()
        duration: float = time.perf_counter() - start
        logger.info("VisualEmbedder loaded in %.2fs on %s", duration, self._device)

    def embed(self, image: Image.Image) -> np.ndarray:
        """Produce a unit-normalized DINOv2 visual embedding.

        Uses the CLS token from the last hidden state as the image-level
        representation. This is the standard DINOv2 retrieval representation.

        Args:
            image: PIL image to embed (any mode; converted to RGB internally).

        Returns:
            1-D float32 numpy array of unit length.
        """
        assert isinstance(image, Image.Image), "image must be a PIL Image"

        rgb_image: Image.Image = image.convert("RGB")
        inputs = self._processor(images=rgb_image, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # CLS token (position 0) is the standard image-level representation for DINOv2
        cls_embedding: np.ndarray = (
            outputs.last_hidden_state[:, 0, :].cpu().float().numpy()[0]
        )
        return normalize_embedding(cls_embedding)
