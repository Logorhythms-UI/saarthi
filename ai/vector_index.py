"""Persistent FAISS vector index for object embeddings.

Wraps a FAISS IndexFlatIP (exact inner-product search on unit-normalized
vectors, which equals cosine similarity) and maintains a parallel mapping
from FAISS row indices to database ObjectObservation primary keys.

Two separate indexes are maintained by the pipeline:

    semantic  — SigLIP2 embeddings (768-dim)
    visual    — DINOv2 embeddings   (384-dim)

Each is stored as a pair of files under ai/indexes/:

    <name>.faiss   — the FAISS index binary
    <name>_ids.json — JSON list mapping FAISS row i → object_id
"""

import json
import logging
import time
from pathlib import Path

import faiss
import numpy as np

logger: logging.Logger = logging.getLogger(__name__)

# Directory for persisted FAISS index files, relative to this file's parent
_INDEX_DIR: Path = Path(__file__).parent / "indexes"


class VectorIndex:
    """Exact cosine-similarity FAISS index with DB-id mapping.

    The model is load-once: the index is loaded from disk on construction
    if files already exist, otherwise a new empty index is created.
    """

    def __init__(self, name: str, dimension: int) -> None:
        """Create or load a named vector index.

        Args:
            name: Logical index name ('semantic' or 'visual'). Used as the
                  filename stem under ai/indexes/.
            dimension: Embedding dimension. Must match the producing model.
        """
        assert isinstance(name, str) and name, "name must be a non-empty string"
        assert (
            isinstance(dimension, int) and dimension > 0
        ), "dimension must be a positive int"

        self._name: str = name
        self._dimension: int = dimension
        self._index_path: Path = _INDEX_DIR / f"{name}.faiss"
        self._ids_path: Path = _INDEX_DIR / f"{name}_ids.json"
        # Parallel list: FAISS row i → database ObjectObservation id
        self._object_ids: list[int] = []

        _INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self._index: faiss.IndexFlatIP = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, embedding: np.ndarray, object_id: int) -> None:
        """Add a single unit-normalized embedding and its DB id to the index.

        Args:
            embedding: 1-D float32 array of length self._dimension.
            object_id: Database ObjectObservation primary key.
        """
        assert isinstance(embedding, np.ndarray), "embedding must be a numpy ndarray"
        assert embedding.shape == (
            self._dimension,
        ), f"embedding must have shape ({self._dimension},), got {embedding.shape}"
        assert (
            isinstance(object_id, int) and object_id > 0
        ), "object_id must be a positive int"

        vector: np.ndarray = embedding.astype(np.float32).reshape(1, -1)
        self._index.add(vector)
        self._object_ids.append(object_id)

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return the k nearest neighbors for a query embedding.

        Args:
            query: 1-D float32 array of length self._dimension.
            k: Maximum number of results to return.

        Returns:
            List of (object_id, score) pairs sorted by descending score.
            May be shorter than k if the index contains fewer entries.
            Returns an empty list if the index is empty.
        """
        assert isinstance(query, np.ndarray), "query must be a numpy ndarray"
        assert query.shape == (
            self._dimension,
        ), f"query must have shape ({self._dimension},), got {query.shape}"
        assert isinstance(k, int) and k > 0, "k must be a positive int"

        if self._index.ntotal == 0:
            logger.info(
                "VectorIndex '%s': index is empty, returning no candidates", self._name
            )
            return []

        effective_k: int = min(k, self._index.ntotal)
        vector: np.ndarray = query.astype(np.float32).reshape(1, -1)
        distances, indices = self._index.search(vector, effective_k)

        results: list[tuple[int, float]] = []
        for faiss_row, score in zip(indices[0], distances[0]):
            if faiss_row < 0:
                # FAISS uses -1 as a sentinel for unfilled slots
                continue
            results.append((self._object_ids[int(faiss_row)], float(score)))

        return results

    def persist(self) -> None:
        """Write the FAISS index and id mapping to disk.

        Called once after a batch of embeddings has been added, not after
        every individual add(), to avoid redundant I/O.
        """
        faiss.write_index(self._index, str(self._index_path))
        with open(self._ids_path, "w", encoding="utf-8") as f:
            json.dump(self._object_ids, f)
        logger.info(
            "VectorIndex '%s': persisted %d vectors to %s",
            self._name,
            self._index.ntotal,
            self._index_path,
        )

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        return int(self._index.ntotal)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> faiss.IndexFlatIP:
        """Load existing index files from disk, or create a fresh empty index."""
        if self._index_path.exists() and self._ids_path.exists():
            start: float = time.perf_counter()
            index: faiss.IndexFlatIP = faiss.read_index(str(self._index_path))
            with open(self._ids_path, "r", encoding="utf-8") as f:
                self._object_ids = json.load(f)
            duration: float = time.perf_counter() - start
            logger.info(
                "VectorIndex '%s': loaded %d vectors from disk in %.2fs",
                self._name,
                index.ntotal,
                duration,
            )
            return index

        logger.info(
            "VectorIndex '%s': no existing index found, creating empty index (dim=%d)",
            self._name,
            self._dimension,
        )
        return faiss.IndexFlatIP(self._dimension)
