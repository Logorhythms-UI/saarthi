"""Application-level orchestration layer.

Coordinates AI models, database, and storage to provide
a clean interface for the UI.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
from ai.broad_detector import BroadDetector, Detection
from ai.visual_embedder import VisualEmbedder
from peewee import IntegrityError
from PIL import Image, UnidentifiedImageError

import config
from ai.object_detector import ObjectDetector
from ai.query_normalizer import QueryNormalizer
from ai.semantic_embedder import SemanticEmbedder
from ai.vector_index import VectorIndex
from db.api import DatabaseEngine
from db.schema import ImageHistory, ObjectObservation
from storage.app import StorageEngine
from utils.result import Result

logger: logging.Logger = logging.getLogger(__name__)

# Embedding dimensions for the two index types.
# Must match the models in SemanticEmbedder and VisualEmbedder.
_SIGLIP2_DIM: int = 768  # google/siglip2-base-patch16-224
_DINOV2_DIM: int = 384  # facebook/dinov2-small


def _calibrate(raw: float, floor: float, ceiling: float) -> float:
    """Map a raw signal value onto [0, 1] using an empirically tuned range.

    `floor` is the raw value expected for an unrelated match, `ceiling` the
    raw value expected for a strong true match (both set in config.py).
    Values below floor clamp to 0.0, above ceiling to 1.0. A degenerate range
    yields 0.0 rather than dividing by zero.
    """
    span: float = ceiling - floor
    if span <= 0.0:
        logger.warning(
            "Invalid calibration range: floor=%.4f ceiling=%.4f", floor, ceiling
        )
        return 0.0
    scaled: float = (raw - floor) / span
    return max(0.0, min(1.0, scaled))


def _compute_final_score(
    semantic_score: float,
    visual_score: float,
    grounding_score: float,
) -> float:
    """Calibrate each raw signal to [0, 1], then combine with config weights.

    Raw SigLIP2 cosine, raw DINOv2 cosine, and Grounding DINO confidence
    have incompatible distributions (see config.py); they must be mapped to
    a common scale before the weighted sum, otherwise MATCH_THRESHOLD
    compares incomparable quantities.
    """
    semantic_cal: float = _calibrate(
        semantic_score, config.SEMANTIC_FLOOR, config.SEMANTIC_CEILING
    )
    visual_cal: float = _calibrate(
        visual_score, config.VISUAL_FLOOR, config.VISUAL_CEILING
    )
    grounding_cal: float = _calibrate(
        grounding_score, config.GROUNDING_FLOOR, config.GROUNDING_CEILING
    )
    return (
        config.SEMANTIC_WEIGHT * semantic_cal
        + config.VISUAL_WEIGHT * visual_cal
        + config.GROUNDING_WEIGHT * grounding_cal
    )


def _iou(box_a: list[float], box_b: list[float]) -> float:
    """Intersection-over-union of two [x1, y1, x2, y2] pixel boxes.

    Returns 0.0 for non-overlapping or degenerate boxes.
    """
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1: float = max(ax1, bx1)
    inter_y1: float = max(ay1, by1)
    inter_x2: float = min(ax2, bx2)
    inter_y2: float = min(ay2, by2)

    inter_w: float = max(0.0, inter_x2 - inter_x1)
    inter_h: float = max(0.0, inter_y2 - inter_y1)
    inter_area: float = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    area_a: float = (ax2 - ax1) * (ay2 - ay1)
    area_b: float = (bx2 - bx1) * (by2 - by1)
    union_area: float = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def _parse_bounding_box(bounding_box_json: str) -> list[float] | None:
    """Parse a stored JSON bounding box into [x1, y1, x2, y2].

    Returns None if the JSON is malformed or does not contain exactly four
    numeric coordinates.
    """
    try:
        box = json.loads(bounding_box_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if (
        not isinstance(box, list)
        or len(box) != 4
        or not all(isinstance(v, (int, float)) for v in box)
    ):
        return None
    return [float(v) for v in box]


def _match_decision(best_score: float, margin: float) -> bool:
    """Return True when the best candidate clears both score and margin gates.

    Kept as a pure function so threshold/margin behaviour is unit-testable
    without loading any models.
    """
    return best_score >= config.MATCH_THRESHOLD and margin >= config.MIN_MARGIN


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
        self._check_index_consistency()
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

            # Grounding DINO verification — 0.0 on failure (not a hard error)
            grounding_score: float = 0.0
            visual_score: float = 0.0

            detect_result: Result = self._object_detector.detect(
                image_path, normalized_query
            )
            if detect_result.ok:
                grounding_score = float(detect_result.result["confidence"])

                # DINOv2 visual consistency, gated by IoU: the visual score only
                # counts when Grounding DINO's verified box overlaps the retrieved
                # object's stored box. Without the gate, "query text appears
                # somewhere in this image" would boost a candidate whose retrieved
                # object is a different thing (e.g. query 'keys' retrieved the
                # remote crop, but keys exist elsewhere in the same photo).
                gd_box: list[float] = detect_result.result["bounding_box"]
                obs_box: list[float] | None = _parse_bounding_box(obs.bounding_box)
                if (
                    obs_box is not None
                    and _iou(gd_box, obs_box) >= config.VISUAL_IOU_GATE
                ):
                    visual_score = self._visual_consistency(
                        image_path, gd_box, int(obs.id)
                    )
            else:
                logger.debug(
                    "Grounding DINO: no detection in %s: %s",
                    Path(image_path).name,
                    detect_result.err,
                )

            final_score: float = _compute_final_score(
                semantic_score=semantic_score,
                visual_score=visual_score,
                grounding_score=grounding_score,
            )

            scored_candidates.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "image_timestamp": str(image_record.timestamp),
                    "label": obs.label,
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

        if not _match_decision(best_score, margin):
            logger.info(
                "Best score %.4f (threshold %.2f), margin %.4f (min %.2f) — NOT_FOUND",
                best_score,
                config.MATCH_THRESHOLD,
                margin,
                config.MIN_MARGIN,
            )
            return Result(
                result=_build_not_found_result(
                    normalized_query=normalized_query,
                    original_query=query,
                    reason=(
                        f"Best candidate score ({best_score:.2f}) is below the "
                        f"confidence threshold ({config.MATCH_THRESHOLD:.2f}), "
                        f"or the margin ({margin:.2f}) is below the minimum "
                        f"({config.MIN_MARGIN:.2f})."
                    ),
                    candidate_count=len(scored_candidates),
                    best_score=best_score,
                    candidates=scored_candidates,
                )
            )

        return Result(
            result=_build_found_result(
                candidate=best,
                normalized_query=normalized_query,
                original_query=query,
                candidate_count=len(scored_candidates),
                margin=margin,
                candidates=scored_candidates,
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

    def delete_image(self, image_id: int) -> Result:
        """Delete an image, its object observations, and its vector embeddings.

        Args:
            image_id: Primary key of the ImageHistory record.

        Returns:
            Result with None on success, or an error message on failure.
        """
        assert isinstance(image_id, int), "image_id must be an integer"

        image_record = self._db.get_image_by_id(image_id)
        if not image_record:
            return Result(err=f"Image with id {image_id} not found")

        image_path = image_record.imgpath

        # 1. Delete from DB and get removed object IDs
        try:
            removed_obs_ids: list[int] = self._db.delete_image(image_id)
        except OSError as e:
            logger.error("Failed to delete image %d from DB: %s", image_id, e)
            return Result(err=f"Database error during deletion: {e}")

        # 2. Remove physical file
        try:
            self._storage.delete_file(image_path)
        except (ValueError, OSError) as e:
            logger.warning("Could not delete physical file %s: %s", image_path, e)
            # Continue anyway since it's removed from DB

        # 3. Remove vectors from FAISS
        if removed_obs_ids:
            try:
                self._semantic_index.remove_objects(removed_obs_ids)
                self._semantic_index.persist()

                self._visual_index.remove_objects(removed_obs_ids)
                self._visual_index.persist()
            except (RuntimeError, ValueError, OSError) as e:
                logger.error("Failed to remove vectors for image %d: %s", image_id, e)
                # Keep going, the DB is the source of truth

        logger.info("Successfully deleted image %d", image_id)
        return Result(result=None)

    def clear_all_memory(self) -> Result:
        """Wipe all images, objects, and embeddings from the system.

        Returns:
            Result with None on success.
        """
        logger.info("Initiating clear all memory...")

        # 1. Clear physical files
        try:
            self._storage.clear_uploads()
        except OSError as e:
            logger.error("Failed to clear upload directory: %s", e)
            return Result(err=f"Failed to clear physical files: {e}")

        # 2. Clear Database
        try:
            self._db.clear_all()
        except OSError as e:
            logger.error("Failed to clear database: %s", e)
            return Result(err=f"Failed to clear database: {e}")

        # 3. Clear Vector Indexes
        try:
            self._semantic_index.clear()
            self._semantic_index.persist()

            self._visual_index.clear()
            self._visual_index.persist()
        except (RuntimeError, ValueError, OSError) as e:
            logger.error("Failed to clear vector indexes: %s", e)
            return Result(err=f"Failed to clear vector indexes: {e}")

        logger.info("Successfully cleared all memory")
        return Result(result=None)

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
            try:
                self._semantic_index.persist()
                self._visual_index.persist()
            except (RuntimeError, OSError) as e:
                # In-memory indexes are correct; only the on-disk copies are
                # stale until the next successful persist. Surfaced, not hidden.
                logger.error(
                    "Vector index persist failed for %s: %s",
                    Path(image_path).name,
                    e,
                )
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

    def _visual_consistency(
        self, image_path: str, gd_box: list[float], object_id: int
    ) -> float:
        """DINOv2 cosine between Grounding DINO's verified region and the
        stored visual embedding of the retrieved object.

        A high score confirms the region just verified is the same physical
        object that was indexed at ingestion time — an independent signal
        from the semantic and grounding channels. Returns 0.0 when the stored
        vector is unavailable or crop/embedding fails.
        """
        assert isinstance(image_path, str), "image_path must be a string"
        assert len(gd_box) == 4, "gd_box must have 4 coordinates"
        assert (
            isinstance(object_id, int) and object_id > 0
        ), "object_id must be a positive int"

        stored: np.ndarray | None = self._visual_index.get_vector(object_id)
        if stored is None:
            logger.warning(
                "Visual vector missing for object %d — visual score 0.0", object_id
            )
            return 0.0

        try:
            with Image.open(image_path) as img:
                rgb: Image.Image = img.convert("RGB")
                crop: Image.Image = rgb.crop(
                    (int(gd_box[0]), int(gd_box[1]), int(gd_box[2]), int(gd_box[3]))
                )
            current: np.ndarray = self._visual_embedder.embed(crop)
        except (UnidentifiedImageError, OSError, ValueError, RuntimeError) as e:
            logger.error(
                "Visual consistency crop/embed failed for %s: %s", image_path, e
            )
            return 0.0

        # Both vectors are unit-normalized, so the dot product is cosine similarity.
        return float(np.dot(current, stored))

    def _check_index_consistency(self) -> None:
        """Warn when DB observation count and index sizes disagree.

        The database is the source of truth; a smaller index means some
        objects are silently un-retrievable until re-ingestion.
        """
        db_count: int = self._db.get_object_observation_count()
        semantic_count: int = self._semantic_index.size
        visual_count: int = self._visual_index.size
        if db_count != semantic_count or db_count != visual_count:
            logger.warning(
                "Index inconsistency: DB has %d observations, semantic index "
                "has %d, visual index has %d. Affected objects cannot be "
                "retrieved; re-ingest images to rebuild.",
                db_count,
                semantic_count,
                visual_count,
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
    candidates: list[dict],
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
        "candidates": candidates,
    }


def _build_not_found_result(
    normalized_query: str,
    original_query: str,
    reason: str,
    candidate_count: int,
    best_score: float | None = None,
    candidates: list[dict] | None = None,
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
    if candidates:
        result["candidates"] = candidates
    return result
