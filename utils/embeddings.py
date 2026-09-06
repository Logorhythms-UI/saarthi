"""Shared embedding helpers."""

import logging

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Return a unit-normalized (L2) copy of an embedding vector."""
    norm: float = float(np.linalg.norm(embedding))
    if norm == 0.0:
        logger.warning("Received zero-norm embedding; returning as-is")
        return embedding.astype(np.float32)
    return (embedding / norm).astype(np.float32)
