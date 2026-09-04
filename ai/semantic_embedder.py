"""Semantic image and text embedder using SigLIP2.

SigLIP2 provides a joint image-text embedding space trained for
cross-modal retrieval. The same model instance handles both image and
text embedding to guarantee embedding-space compatibility.

Usage in the pipeline:

    Ingestion:  object crop → embed_image() → semantic embedding
    Search:     normalized query → embed_text() → query embedding
"""

import logging
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

logger: logging.Logger = logging.getLogger(__name__)

# Smallest SigLIP2 variant available through HuggingFace transformers.
# Larger variants (384, 512) provide higher quality at increased memory cost.
_MODEL_ID: str = "google/siglip2-base-patch16-224"


class SemanticEmbedder:
    """Produces normalized SigLIP2 embeddings for images and text.

    The model is loaded once and remains in memory for the application lifetime.
    Embeddings are unit-normalized for use with cosine similarity (FAISS IndexFlatIP).
    """

    def __init__(self) -> None:
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading SemanticEmbedder (SigLIP2) on device: %s", self._device)

        start: float = time.perf_counter()
        self._processor: AutoProcessor = AutoProcessor.from_pretrained(_MODEL_ID)
        self._model: AutoModel = AutoModel.from_pretrained(_MODEL_ID).to(self._device)
        self._model.eval()
        duration: float = time.perf_counter() - start
        logger.info("SemanticEmbedder loaded in %.2fs on %s", duration, self._device)

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Produce a unit-normalized SigLIP2 image embedding.

        Args:
            image: PIL image to embed (any mode; converted to RGB internally).

        Returns:
            1-D float32 numpy array of unit length.
        """
        assert isinstance(image, Image.Image), "image must be a PIL Image"

        rgb_image: Image.Image = image.convert("RGB")
        inputs = self._processor(images=rgb_image, return_tensors="pt").to(self._device)

        with torch.no_grad():
            raw: torch.Tensor = self._model.get_image_features(**inputs)

        return _normalize(raw.cpu().float().numpy()[0])

    def embed_text(self, text: str) -> np.ndarray:
        """Produce a unit-normalized SigLIP2 text embedding.

        Args:
            text: Non-empty text string to embed.

        Returns:
            1-D float32 numpy array of unit length.
        """
        assert isinstance(text, str), "text must be a string"
        assert len(text.strip()) > 0, "text must not be empty"

        # padding="max_length" is required by the SigLIP tokenizer
        inputs = self._processor(
            text=text, return_tensors="pt", padding="max_length"
        ).to(self._device)

        with torch.no_grad():
            raw: torch.Tensor = self._model.get_text_features(**inputs)

        return _normalize(raw.cpu().float().numpy()[0])


def _normalize(embedding: np.ndarray) -> np.ndarray:
    """Return a unit-normalized (L2) copy of the embedding vector.

    Returns the vector unchanged if its norm is zero to avoid
    division by zero on degenerate model outputs.
    """
    norm: float = float(np.linalg.norm(embedding))
    if norm == 0.0:
        logger.warning(
            "SemanticEmbedder: received zero-norm embedding — returning as-is"
        )
        return embedding.astype(np.float32)
    return (embedding / norm).astype(np.float32)
