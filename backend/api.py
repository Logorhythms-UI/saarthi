"""Application-level orchestration layer.

Coordinates AI models, database, and storage to provide
a clean interface for the UI.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
from peewee import IntegrityError
from PIL import Image, UnidentifiedImageError

import config
from ai.broad_detector import BroadDetector, Detection
from ai.object_detector import ObjectDetector
from ai.query_normalizer import QueryNormalizer
from ai.semantic_embedder import SemanticEmbedder
from ai.vector_index import VectorIndex
from ai.visual_embedder import VisualEmbedder
from db.api import DatabaseEngine
from db.schema import ImageHistory, ObjectObservation
from storage.app import StorageEngine
from utils.result import Result

logger: logging.Logger = logging.getLogger(__name__)

# Embedding dimensions for the two index types.
# Must match the models in SemanticEmbedder and VisualEmbedder.
_SIGLIP2_DIM: int = 768  # google/siglip2-base-patch16-224
_DINOV2_DIM: int = 384  # facebook/dinov2-small


def _compute_final_score(
    semantic_score: float,
    visual_score: float,
    grounding_score: float,
) -> float:
    """Combine per-signal scores into a single weighted final score.

    Weights are read from config so they can be tuned without code changes.
    Individual signals remain separate so they can be inspected for debugging.
    """
    return (
        config.SEMANTIC_WEIGHT * semantic_score
        + config.VISUAL_WEIGHT * visual_score
        + config.GROUNDING_WEIGHT * grounding_score
    )


class Backend:
    """Application-level orchestration layer.

    Coordinates AI models, database, and storage to provide
    a clean interface for the UI.
    """

    def __init__(self) -> None:
        self._storage: StorageEngine = StorageEngine()
        self._db: DatabaseEngine = DatabaseEngine()
        self._query_normalizer: QueryNormalizer = QueryNormalizer()
        self._object_detector: ObjectDetector = ObjectDetector()
        self._broad_detector: BroadDetector = BroadDetector()
        self._semantic_embedder: SemanticEmbedder = SemanticEmbedder()
        self._visual_embedder: VisualEmbedder = VisualEmbedder()
        self._semantic_index: VectorIndex = VectorIndex("semantic", _SIGLIP2_DIM)
        self._visual_index: VectorIndex = VectorIndex("visual", _DINOV2_DIM)
        logger.info("Backend initialized — all subsystems ready")

    def ingest_upload(self, file_content: bytes, filename: str) -> Result:
        """Store an uploaded image, register it in the database, and index its objects.

        Accepts raw file bytes (from NiceGUI upload handler).
        Returns Result with the stored path on success.

        Object indexing failures are logged but do NOT cause this method to
        fail — the image is already safely stored and registered.
        """
        assert isinstance(file_content, bytes), "file_content must be bytes"
        assert isinstance(filename, str), "filename must be a string"

        try:
            stored_path: str = self._storage.store_upload(file_content, filename)
        except (AssertionError, ValueError) as e:
            logger.error("Storage rejected upload '%s': %s", filename, e)
            return Result(err=str(e))
        except OSError as e:
            logger.error("Failed to store upload '%s': %s", filename, e)
            return Result(err=f"Failed to store image: {e}")

        try:
            image_record: ImageHistory = self._db.add_image(stored_path)
        except IntegrityError:
            logger.info("Image already exists in database: %s", stored_path)
            return Result(result=stored_path)
        except (AssertionError, OSError) as e:
            logger.error("Database rejected image '%s': %s", stored_path, e)
            return Result(err=str(e))

        self._ingest_objects(stored_path, image_record)
        logger.info("Ingested upload: %s", filename)
        return Result(result=stored_path)

    def ingest_image(self, image_path: str) -> Result:
        """Store a single image from a file path, register in DB, and index its objects.

        Returns Result with the stored path on success.

        Object indexing failures are logged but do NOT cause this method to
        fail — the image is already safely stored and registered.
        """
        assert isinstance(image_path, str), "image_path must be a string"

        try:
            stored_path: str = self._storage.upload_image(image_path)
        except AssertionError as e:
            logger.error("Invalid image path %s: %s", image_path, e)
            return Result(err=str(e))
        except (FileNotFoundError, OSError) as e:
            logger.error("Failed to store image %s: %s", image_path, e)
            return Result(err=str(e))

        try:
            image_record: ImageHistory = self._db.add_image(stored_path)
        except IntegrityError:
            logger.info("Image already exists in database: %s", stored_path)
            return Result(result=stored_path)
        except (AssertionError, OSError) as e:
            logger.error("Database rejected image %s: %s", stored_path, e)
            return Result(err=str(e))

        self._ingest_objects(stored_path, image_record)
        logger.info("Ingested image: %s", stored_path)
        return Result(result=stored_path)

    def search(self, query: str) -> Result:
        """Execute the full retrieval pipeline for a natural-language query.

        Pipeline:
            1. SLM query normalisation
            2. SigLIP2 text embedding
            3. Semantic FAISS top-K retrieval
            4. Collapse object candidates to image candidates
            5. Grounding DINO verification of each image candidate
            6. Final scoring + threshold decision

        Returns Result containing a dict with key 'status' set to either
        'FOUND' or 'NOT_FOUND'.  See _build_found_result and
        _build_not_found_result for the full field sets.

        Operational errors (model failure, etc.) are returned as
        Result(err=...) — a distinct outcome from NOT_FOUND.
        """
        assert isinstance(query, str), "query must be a string"
        assert len(query.strip()) > 0, "query must not be empty"

        total_start: float = time.perf_counter()
        logger.info("Search pipeline started — query: '%s'", query)

        # Phase 1: Query normalisation via SLM
        normalize_result: Result = self._query_normalizer.normalize(query)
        slm_duration: float = time.perf_counter() - total_start

        if not normalize_result.ok:
            logger.error("Query normalisation failed: %s", normalize_result.err)
            return Result(err=f"Query processing failed: {normalize_result.err}")

        normalized_query: str = normalize_result.result
        logger.info(
            "SLM normalisation completed in %.2fs: '%s' → '%s'",
            slm_duration,
            query,
            normalized_query,
        )

        # Phase 2: Embed the normalized query with SigLIP2
        try:
            query_embedding: np.ndarray = self._semantic_embedder.embed_text(
                normalized_query
            )
        except (ValueError, RuntimeError) as e:
            logger.error("Query embedding failed: %s", e)
            return Result(err=f"Query embedding failed: {e}")

        # Phase 3: Semantic FAISS retrieval — top-K object candidates
        search_start: float = time.perf_counter()
        raw_candidates: list[tuple[int, float]] = self._semantic_index.search(
            query_embedding, config.SEMANTIC_TOP_K
        )

        if not raw_candidates:
            logger.info("Semantic index returned no candidates (index may be empty)")
            return Result(
                result=_build_not_found_result(
                    normalized_query=normalized_query,
                    original_query=query,
                    reason="No objects have been indexed yet. Upload images first.",
                    candidate_count=0,
                )
            )

        logger.info("Semantic retrieval: %d raw object candidates", len(raw_candidates))

        # Phase 4: Resolve object ids to DB records and collapse to image candidates.
        # For each image, retain the single best-scoring object candidate.
        object_ids: list[int] = [obj_id for obj_id, _ in raw_candidates]
        score_by_id: dict[int, float] = {
            obj_id: score for obj_id, score in raw_candidates
        }
        observations: list[ObjectObservation] = self._db.get_object_observations_by_ids(
            object_ids
        )

        # image_id → (object_observation, semantic_score)
        best_per_image: dict[int, tuple[ObjectObservation, float]] = {}
        for obs in observations:
            image_id: int = int(obs.image_id)
            score: float = score_by_id.get(int(obs.id), 0.0)
            if image_id not in best_per_image or score > best_per_image[image_id][1]:
                best_per_image[image_id] = (obs, score)

        logger.info(
            "Collapsed %d object candidates to %d unique image candidates",
            len(raw_candidates),
            len(best_per_image),
        )

        # Phase 5: Verify each image candidate with Grounding DINO and score.
        scored_candidates: list[dict] = []

        for image_id, (obs, semantic_score) in best_per_image.items():
            image_record: ImageHistory | None = self._db.get_image_by_id(image_id)
            if image_record is None:
                logger.warning(
                    "Object observation %d references missing image_id %d — skipping",
                    obs.id,
                    image_id,
                )
                continue

            image_path: str = image_record.imgpath
            if not Path(image_path).exists():
                logger.warning(
                    "Image file missing from storage: %s — skipping", image_path
                )
                continue

            # Grounding DINO verification — score 0.0 on failure (not a hard error)
            grounding_score: float = 0.0
            detect_result: Result = self._object_detector.detect(
                image_path, normalized_query
            )
            if detect_result.ok:
                grounding_score = float(detect_result.result["confidence"])
            else:
                logger.debug(
                    "Grounding DINO: no detection in %s: %s",
                    Path(image_path).name,
                    detect_result.err,
                )

            # DINOv2 visual_score: deferred — no valid text→DINOv2 query mechanism.
            # Kept as a named, zero-weighted signal so the pipeline structure is
            # preserved and the score can be wired in when the design is resolved.
            visual_score: float = 0.0

            final_score: float = _compute_final_score(
                semantic_score=max(0.0, semantic_score),
                visual_score=visual_score,
                grounding_score=grounding_score,
            )

            scored_candidates.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "image_timestamp": str(image_record.timestamp),
                    "semantic_score": round(semantic_score, 4),
                    "visual_score": round(visual_score, 4),
                    "grounding_score": round(grounding_score, 4),
                    "final_score": round(final_score, 4),
                }
            )

        search_duration: float = time.perf_counter() - search_start
        total_duration: float = time.perf_counter() - total_start

        if not scored_candidates:
            logger.info(
                "All candidates were unresolvable (%.2fs total)", total_duration
            )
            return Result(
                result=_build_not_found_result(
                    normalized_query=normalized_query,
                    original_query=query,
                    reason="Retrieved candidates could not be resolved to stored images.",
                    candidate_count=0,
                )
            )

        # Phase 6: Rank and apply threshold
        scored_candidates.sort(key=lambda c: c["final_score"], reverse=True)
        best: dict = scored_candidates[0]
        best_score: float = best["final_score"]
        second_score: float = (
            scored_candidates[1]["final_score"] if len(scored_candidates) > 1 else 0.0
        )
        margin: float = round(best_score - second_score, 4)

        logger.info(
            "Search complete — %d candidates scored, best=%.4f, margin=%.4f "
            "(threshold=%.2f), search=%.2fs, total=%.2fs",
            len(scored_candidates),
            best_score,
            margin,
            config.MATCH_THRESHOLD,
            search_duration,
            total_duration,
        )

        if best_score < config.MATCH_THRESHOLD:
            logger.info(
                "Best score %.4f below threshold %.2f — NOT_FOUND",
                best_score,
                config.MATCH_THRESHOLD,
            )
            return Result(
                result=_build_not_found_result(
                    normalized_query=normalized_query,
                    original_query=query,
                    reason=(
                        f"Best candidate score ({best_score:.2f}) is below the "
                        f"confidence threshold ({config.MATCH_THRESHOLD:.2f})."
                    ),
                    candidate_count=len(scored_candidates),
                    best_score=best_score,
                )
            )

        return Result(
            result=_build_found_result(
                candidate=best,
                normalized_query=normalized_query,
                original_query=query,
                candidate_count=len(scored_candidates),
                margin=margin,
                slm_duration_s=round(slm_duration, 2),
                search_duration_s=round(search_duration, 2),
            )
        )

    def get_image_count(self) -> int:
        """Return the number of images in the database."""
        return self._db.get_image_count()

    def get_all_images_desc(self) -> list[ImageHistory]:
        """Return all image records ordered newest-first by sequence number."""
        return self._db.get_all_images_desc()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ingest_objects(self, image_path: str, image_record: ImageHistory) -> None:
        """Detect objects in image_path and add them to the FAISS indexes.

        Failures at any sub-step are logged at ERROR level but do NOT
        propagate — the caller (ingest_upload / ingest_image) has already
        persisted the image and committed the DB record.

        Persistence (persist()) is called once after all objects are added
        to avoid redundant I/O on a per-object basis.
        """
        logger.info("Object ingestion started for: %s", Path(image_path).name)

        try:
            pil_image: Image.Image = Image.open(image_path).convert("RGB")
        except UnidentifiedImageError as e:
            logger.error("Cannot open image for object ingestion %s: %s", image_path, e)
            return
        except (FileNotFoundError, OSError) as e:
            logger.error("Image missing during object ingestion %s: %s", image_path, e)
            return

        detections: list[Detection] = self._broad_detector.detect(pil_image)
        logger.info(
            "Object ingestion: %d object(s) detected in %s",
            len(detections),
            Path(image_path).name,
        )

        if not detections:
            return

        img_width: int = pil_image.width
        img_height: int = pil_image.height
        added: int = 0

        for detection in detections:
            x1, y1, x2, y2 = detection.bounding_box
            # Clamp to image bounds
            x1 = max(0.0, x1)
            y1 = max(0.0, y1)
            x2 = min(float(img_width), x2)
            y2 = min(float(img_height), y2)

            if x2 <= x1 or y2 <= y1:
                logger.warning(
                    "Skipping degenerate bounding box [%.1f, %.1f, %.1f, %.1f]",
                    x1,
                    y1,
                    x2,
                    y2,
                )
                continue

            crop: Image.Image = pil_image.crop((x1, y1, x2, y2))

            try:
                semantic_emb: np.ndarray = self._semantic_embedder.embed_image(crop)
            except (ValueError, RuntimeError) as e:
                logger.error(
                    "Semantic embedding failed for object '%s' in %s: %s",
                    detection.label,
                    Path(image_path).name,
                    e,
                )
                continue

            try:
                visual_emb: np.ndarray = self._visual_embedder.embed(crop)
            except (ValueError, RuntimeError) as e:
                logger.error(
                    "Visual embedding failed for object '%s' in %s: %s",
                    detection.label,
                    Path(image_path).name,
                    e,
                )
                continue

            bounding_box_json: str = json.dumps([x1, y1, x2, y2])

            try:
                object_id: int = self._db.add_object_observation(
                    image_id=int(image_record.id),
                    label=detection.label,
                    detector_confidence=detection.confidence,
                    bounding_box=bounding_box_json,
                    timestamp=image_record.timestamp,
                )
            except (AssertionError, OSError) as e:
                logger.error(
                    "DB insert failed for object '%s' in %s: %s",
                    detection.label,
                    Path(image_path).name,
                    e,
                )
                continue

            self._semantic_index.add(semantic_emb, object_id)
            self._visual_index.add(visual_emb, object_id)
            added += 1

        if added > 0:
            self._semantic_index.persist()
            self._visual_index.persist()
            logger.info(
                "Object ingestion complete: %d/%d object(s) indexed for %s",
                added,
                len(detections),
                Path(image_path).name,
            )
        else:
            logger.warning(
                "Object ingestion: no objects successfully indexed for %s",
                Path(image_path).name,
            )


# ------------------------------------------------------------------
# Result builders — keep search() readable by isolating dict construction
# ------------------------------------------------------------------


def _build_found_result(
    candidate: dict,
    normalized_query: str,
    original_query: str,
    candidate_count: int,
    margin: float,
    slm_duration_s: float,
    search_duration_s: float,
) -> dict:
    """Build the result dict for a successful FOUND outcome."""
    return {
        "status": "FOUND",
        "image_path": candidate["image_path"],
        "normalized_query": normalized_query,
        "original_query": original_query,
        "image_timestamp": candidate["image_timestamp"],
        "final_score": candidate["final_score"],
        "semantic_score": candidate["semantic_score"],
        "visual_score": candidate["visual_score"],
        "grounding_score": candidate["grounding_score"],
        "candidate_count": candidate_count,
        "margin": margin,
        "slm_duration_s": slm_duration_s,
        "search_duration_s": search_duration_s,
    }


def _build_not_found_result(
    normalized_query: str,
    original_query: str,
    reason: str,
    candidate_count: int,
    best_score: float | None = None,
) -> dict:
    """Build the result dict for a NOT_FOUND outcome."""
    result: dict = {
        "status": "NOT_FOUND",
        "normalized_query": normalized_query,
        "original_query": original_query,
        "reason": reason,
        "candidate_count": candidate_count,
    }
    if best_score is not None:
        result["best_score"] = best_score
    return result
