"""Unit tests for the MVP object-centric retrieval pipeline.

All ML models (BroadDetector, SemanticEmbedder, VisualEmbedder, ObjectDetector,
QueryNormalizer) are mocked so tests run without GPU, model weights, or network
access.  The database uses an isolated in-memory SQLite instance via a temporary
file so tests never touch the production database.db.

The tests cover:
    - VectorIndex: add, search, persistence, empty-index behaviour
    - Object ingestion: detection → observations → indexes
    - Multiple objects per image → multiple observations
    - Query retrieval: controlled embedding retrieves expected candidate
    - Duplicate image candidates: collapse to one image candidate
    - Candidate ranking: weighted score selects correct candidate
    - False-positive rejection: NOT_FOUND when all scores below threshold
    - Grounding DINO false-positive regression: correct image wins despite
      a higher Grounding DINO confidence in the wrong image
    - Object absent: NOT_FOUND when object is not in memory
    - Empty index: clean NOT_FOUND on empty semantic index
"""

import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vector(size: int, seed: int) -> np.ndarray:
    """Produce a reproducible unit-normalized vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(size).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_tiny_image() -> Image.Image:
    """Create a 32×32 RGB PIL image suitable for testing."""
    return Image.new("RGB", (32, 32), color=(100, 150, 200))


# ---------------------------------------------------------------------------
# VectorIndex tests (no mocking needed — pure FAISS behaviour)
# ---------------------------------------------------------------------------


class TestVectorIndex:
    """Tests for ai.vector_index.VectorIndex."""

    def test_empty_index_returns_no_candidates(self, tmp_path, monkeypatch):
        """An empty index returns an empty list from search()."""
        monkeypatch.setattr("ai.vector_index._INDEX_DIR", tmp_path)
        from ai.vector_index import VectorIndex

        idx = VectorIndex("test_empty", 64)
        query = _unit_vector(64, seed=0)
        results = idx.search(query, k=5)
        assert results == []

    def test_add_and_search(self, tmp_path, monkeypatch):
        """Added vectors are found by search with the correct object_id."""
        monkeypatch.setattr("ai.vector_index._INDEX_DIR", tmp_path)
        from ai.vector_index import VectorIndex

        dim = 64
        idx = VectorIndex("test_search", dim)
        target = _unit_vector(dim, seed=42)
        noise = _unit_vector(dim, seed=99)

        idx.add(target, object_id=1)
        idx.add(noise, object_id=2)

        results = idx.search(target, k=2)
        assert len(results) == 2
        # Top result must be the exact match
        top_id, top_score = results[0]
        assert top_id == 1
        assert top_score == pytest.approx(1.0, abs=1e-5)

    def test_persist_and_reload(self, tmp_path, monkeypatch):
        """After persist(), a fresh VectorIndex loads vectors from disk."""
        monkeypatch.setattr("ai.vector_index._INDEX_DIR", tmp_path)
        from ai.vector_index import VectorIndex

        dim = 64
        v = _unit_vector(dim, seed=7)

        idx = VectorIndex("test_persist", dim)
        idx.add(v, object_id=5)
        idx.persist()

        # Reload — must find the previously added vector
        idx2 = VectorIndex("test_persist", dim)
        assert idx2.size == 1
        results = idx2.search(v, k=1)
        assert results[0][0] == 5

    def test_size_property(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ai.vector_index._INDEX_DIR", tmp_path)
        from ai.vector_index import VectorIndex

        idx = VectorIndex("test_size", 64)
        assert idx.size == 0
        idx.add(_unit_vector(64, seed=1), object_id=10)
        assert idx.size == 1


# ---------------------------------------------------------------------------
# Database layer tests (isolated temp DB)
# ---------------------------------------------------------------------------


@pytest.fixture()
def temp_db(tmp_path):
    """Provide an isolated DatabaseEngine backed by a temp SQLite file."""
    import db.schema as schema_module

    original_db = schema_module.db

    test_db_path = str(tmp_path / "test.db")
    from peewee import SqliteDatabase

    test_db = SqliteDatabase(
        test_db_path,
        pragmas={"journal_mode": "wal", "foreign_keys": 1},
        check_same_thread=False,
    )

    # Patch the module-level db object and both models' Meta.database
    schema_module.db = test_db
    schema_module.ImageHistory._meta.database = test_db
    schema_module.ObjectObservation._meta.database = test_db

    test_db.connect()
    test_db.create_tables(
        [schema_module.ImageHistory, schema_module.ObjectObservation], safe=True
    )
    test_db.close()

    from db.api import DatabaseEngine

    engine = DatabaseEngine()

    yield engine

    # Restore original db
    schema_module.db = original_db
    schema_module.ImageHistory._meta.database = original_db
    schema_module.ObjectObservation._meta.database = original_db


class TestDatabaseEngine:
    def test_add_image_returns_record(self, temp_db, tmp_path):
        img = tmp_path / "a.jpg"
        img.write_bytes(b"fake")
        record = temp_db.add_image(str(img))
        from db.schema import ImageHistory

        assert isinstance(record, ImageHistory)
        assert record.imgpath == str(img)

    def test_add_object_observation_returns_id(self, temp_db, tmp_path):
        img = tmp_path / "b.jpg"
        img.write_bytes(b"fake")
        image_record = temp_db.add_image(str(img))

        obs_id = temp_db.add_object_observation(
            image_id=int(image_record.id),
            label="cup",
            detector_confidence=0.8,
            bounding_box="[0,0,10,10]",
            timestamp=datetime.datetime.now(datetime.UTC),
        )
        assert isinstance(obs_id, int)
        assert obs_id > 0

    def test_get_object_observations_by_ids(self, temp_db, tmp_path):
        img = tmp_path / "c.jpg"
        img.write_bytes(b"fake")
        image_record = temp_db.add_image(str(img))

        oid1 = temp_db.add_object_observation(
            int(image_record.id),
            "chair",
            0.7,
            "[0,0,5,5]",
            datetime.datetime.now(datetime.UTC),
        )
        oid2 = temp_db.add_object_observation(
            int(image_record.id),
            "table",
            0.6,
            "[5,5,10,10]",
            datetime.datetime.now(datetime.UTC),
        )

        obs = temp_db.get_object_observations_by_ids([oid1, oid2])
        assert len(obs) == 2
        labels = {o.label for o in obs}
        assert labels == {"chair", "table"}

    def test_get_object_observations_empty_ids(self, temp_db):
        assert temp_db.get_object_observations_by_ids([]) == []

    def test_get_image_by_id(self, temp_db, tmp_path):
        img = tmp_path / "d.jpg"
        img.write_bytes(b"fake")
        record = temp_db.add_image(str(img))
        fetched = temp_db.get_image_by_id(int(record.id))
        assert fetched is not None
        assert fetched.imgpath == str(img)

    def test_get_image_by_id_missing(self, temp_db):
        assert temp_db.get_image_by_id(99999) is None


# ---------------------------------------------------------------------------
# Backend pipeline tests (models fully mocked)
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_backend(tmp_path, monkeypatch):
    """Return a Backend instance with all AI models and storage mocked."""
    # Patch index storage directory
    monkeypatch.setattr("ai.vector_index._INDEX_DIR", tmp_path / "indexes")

    # Patch DB to use temp file
    from peewee import SqliteDatabase

    import db.schema as schema_module

    test_db = SqliteDatabase(
        str(tmp_path / "pipeline_test.db"),
        pragmas={"journal_mode": "wal", "foreign_keys": 1},
        check_same_thread=False,
    )
    schema_module.db = test_db
    schema_module.ImageHistory._meta.database = test_db
    schema_module.ObjectObservation._meta.database = test_db

    test_db.connect()
    test_db.create_tables(
        [schema_module.ImageHistory, schema_module.ObjectObservation], safe=True
    )
    test_db.close()

    # Patch StorageEngine to just return the provided path unchanged
    mock_storage = MagicMock()
    mock_storage.store_upload.side_effect = lambda content, name: str(tmp_path / name)
    mock_storage.upload_image.side_effect = lambda p: p

    # Mocked AI components
    mock_broad = MagicMock()
    mock_semantic = MagicMock()
    mock_visual = MagicMock()
    mock_grounding = MagicMock()
    mock_normalizer = MagicMock()

    with (
        patch("backend.api.StorageEngine", return_value=mock_storage),
        patch("backend.api.BroadDetector", return_value=mock_broad),
        patch("backend.api.SemanticEmbedder", return_value=mock_semantic),
        patch("backend.api.VisualEmbedder", return_value=mock_visual),
        patch("backend.api.ObjectDetector", return_value=mock_grounding),
        patch("backend.api.QueryNormalizer", return_value=mock_normalizer),
    ):
        from backend.api import Backend

        backend = Backend()

    # Attach mocks for test-level configuration
    backend._mock_storage = mock_storage
    backend._mock_broad = mock_broad
    backend._mock_semantic = mock_semantic
    backend._mock_visual = mock_visual
    backend._mock_grounding = mock_grounding
    backend._mock_normalizer = mock_normalizer

    yield backend, tmp_path

    # Restore original DB
    from db.schema import db as original_db

    schema_module.ImageHistory._meta.database = original_db
    schema_module.ObjectObservation._meta.database = original_db
    schema_module.db = original_db


def _write_fake_image(path: Path) -> None:
    """Write a minimal valid JPEG-like file to path."""
    img = Image.new("RGB", (32, 32), (128, 128, 128))
    img.save(str(path), format="JPEG")


class TestIngestionPipeline:
    def test_single_object_creates_observation(self, mock_backend):
        """One detected object → one ObjectObservation in the DB + FAISS."""
        backend, tmp_path = mock_backend
        img_path = tmp_path / "img1.jpg"
        _write_fake_image(img_path)

        from ai.broad_detector import Detection

        backend._mock_broad.detect.return_value = [
            Detection(label="bag", confidence=0.9, bounding_box=[0, 0, 16, 16])
        ]
        sem_vec = _unit_vector(768, seed=1)
        vis_vec = _unit_vector(384, seed=2)
        backend._mock_semantic.embed_image.return_value = sem_vec
        backend._mock_visual.embed.return_value = vis_vec

        result = backend.ingest_image(str(img_path))
        assert result.ok

        # FAISS index must have exactly 1 vector
        assert backend._semantic_index.size == 1
        assert backend._visual_index.size == 1

        # DB must have exactly 1 ObjectObservation
        obs = backend._db.get_object_observations_by_ids([1])
        assert len(obs) == 1
        assert obs[0].label == "bag"

    def test_multiple_objects_create_multiple_observations(self, mock_backend):
        """Two detected objects → two ObjectObservations."""
        backend, tmp_path = mock_backend
        img_path = tmp_path / "img2.jpg"
        _write_fake_image(img_path)

        from ai.broad_detector import Detection

        backend._mock_broad.detect.return_value = [
            Detection(label="cup", confidence=0.8, bounding_box=[0, 0, 10, 10]),
            Detection(label="book", confidence=0.7, bounding_box=[10, 10, 20, 20]),
        ]
        backend._mock_semantic.embed_image.return_value = _unit_vector(768, seed=3)
        backend._mock_visual.embed.return_value = _unit_vector(384, seed=4)

        result = backend.ingest_image(str(img_path))
        assert result.ok
        assert backend._semantic_index.size == 2

    def test_no_detections_does_not_error(self, mock_backend):
        """An image with no broad detections ingests cleanly — no FAISS entries."""
        backend, tmp_path = mock_backend
        img_path = tmp_path / "img3.jpg"
        _write_fake_image(img_path)

        backend._mock_broad.detect.return_value = []

        result = backend.ingest_image(str(img_path))
        assert result.ok
        assert backend._semantic_index.size == 0

    def test_crops_are_not_saved_to_disk(self, mock_backend, tmp_path):
        """Object crops must remain in memory — no extra files should appear."""
        backend, tmp_path = mock_backend
        img_path = tmp_path / "img4.jpg"
        _write_fake_image(img_path)

        from ai.broad_detector import Detection

        backend._mock_broad.detect.return_value = [
            Detection(label="key", confidence=0.85, bounding_box=[0, 0, 16, 16])
        ]
        backend._mock_semantic.embed_image.return_value = _unit_vector(768, seed=5)
        backend._mock_visual.embed.return_value = _unit_vector(384, seed=6)

        before_files = set(tmp_path.iterdir())
        backend.ingest_image(str(img_path))
        after_files = set(tmp_path.iterdir())

        # Only the source image should appear — no crop files
        new_files = after_files - before_files
        # The index dir may be created; exclude it and .db file
        new_non_dir = {f for f in new_files if f.is_file() and f.suffix != ".db"}
        assert new_non_dir == set(), f"Unexpected crop files on disk: {new_non_dir}"


class TestSearchPipeline:
    def _ingest_image_with_embedding(
        self, backend, tmp_path, name: str, semantic_vec: np.ndarray, seed_vis: int
    ) -> str:
        """Helper: ingest a single-object image with a known semantic embedding."""
        img_path = tmp_path / name
        _write_fake_image(img_path)

        from ai.broad_detector import Detection

        backend._mock_broad.detect.return_value = [
            Detection(label="object", confidence=0.9, bounding_box=[0, 0, 16, 16])
        ]
        backend._mock_semantic.embed_image.return_value = semantic_vec
        backend._mock_visual.embed.return_value = _unit_vector(384, seed=seed_vis)

        backend.ingest_image(str(img_path))
        return str(img_path)

    def test_query_retrieves_expected_candidate(self, mock_backend):
        """Controlled query embedding retrieves the matching image."""
        backend, tmp_path = mock_backend

        target_vec = _unit_vector(768, seed=10)
        noise_vec = _unit_vector(768, seed=11)

        target_path = self._ingest_image_with_embedding(
            backend, tmp_path, "target.jpg", target_vec, seed_vis=20
        )
        self._ingest_image_with_embedding(
            backend, tmp_path, "noise.jpg", noise_vec, seed_vis=21
        )

        from utils.result import Result

        backend._mock_normalizer.normalize.return_value = Result(result="bag")
        backend._mock_semantic.embed_text.return_value = target_vec
        backend._mock_grounding.detect.return_value = Result(
            result={"label": "bag", "confidence": 0.9, "bounding_box": [0, 0, 16, 16]}
        )

        result = backend.search("where is my bag")
        assert result.ok
        assert result.result["status"] == "FOUND"
        assert result.result["image_path"] == target_path

    def test_duplicate_image_candidates_collapse(self, mock_backend):
        """Two objects from the same image produce only one image candidate."""
        backend, tmp_path = mock_backend

        img_path = tmp_path / "multi.jpg"
        _write_fake_image(img_path)

        from ai.broad_detector import Detection

        vec_a = _unit_vector(768, seed=30)
        vec_b = _unit_vector(768, seed=31)

        backend._mock_broad.detect.return_value = [
            Detection(label="cup", confidence=0.8, bounding_box=[0, 0, 10, 10]),
            Detection(label="plate", confidence=0.7, bounding_box=[10, 10, 20, 20]),
        ]
        backend._mock_semantic.embed_image.side_effect = [vec_a, vec_b]
        backend._mock_visual.embed.return_value = _unit_vector(384, seed=32)
        backend.ingest_image(str(img_path))

        from utils.result import Result

        query_vec = _unit_vector(768, seed=30)  # matches vec_a
        backend._mock_normalizer.normalize.return_value = Result(result="cup")
        backend._mock_semantic.embed_text.return_value = query_vec
        backend._mock_grounding.detect.return_value = Result(
            result={"label": "cup", "confidence": 0.8, "bounding_box": [0, 0, 10, 10]}
        )

        result = backend.search("cup")
        assert result.ok
        data = result.result
        # Only one image candidate should have been evaluated
        assert data["candidate_count"] == 1

    def test_ranking_selects_highest_scoring_candidate(self, mock_backend):
        """Given two images, the one with higher final_score is returned."""
        backend, tmp_path = mock_backend

        vec_strong = _unit_vector(768, seed=40)
        vec_weak = _unit_vector(768, seed=41)

        # Ensure vec_strong will score higher than vec_weak against the query
        query_vec = vec_strong.copy()

        strong_path = self._ingest_image_with_embedding(
            backend, tmp_path, "strong.jpg", vec_strong, seed_vis=50
        )
        self._ingest_image_with_embedding(
            backend, tmp_path, "weak.jpg", vec_weak, seed_vis=51
        )

        from utils.result import Result

        backend._mock_normalizer.normalize.return_value = Result(result="keys")
        backend._mock_semantic.embed_text.return_value = query_vec
        # Grounding: strong image gets 0.9, weak gets 0.2
        backend._mock_grounding.detect.side_effect = [
            Result(
                result={
                    "label": "keys",
                    "confidence": 0.9,
                    "bounding_box": [0, 0, 5, 5],
                }
            ),
            Result(
                result={
                    "label": "keys",
                    "confidence": 0.2,
                    "bounding_box": [0, 0, 5, 5],
                }
            ),
        ]

        result = backend.search("keys")
        assert result.ok
        assert result.result["status"] == "FOUND"
        assert result.result["image_path"] == strong_path

    def test_not_found_when_below_threshold(self, mock_backend, monkeypatch):
        """If the best score is below MATCH_THRESHOLD, NOT_FOUND is returned."""
        import config

        monkeypatch.setattr(config, "MATCH_THRESHOLD", 0.99)  # impossibly high

        backend, tmp_path = mock_backend
        vec = _unit_vector(768, seed=60)
        self._ingest_image_with_embedding(
            backend, tmp_path, "img.jpg", vec, seed_vis=61
        )

        from utils.result import Result

        query_vec = _unit_vector(768, seed=62)  # different from stored vec
        backend._mock_normalizer.normalize.return_value = Result(result="bottle")
        backend._mock_semantic.embed_text.return_value = query_vec
        backend._mock_grounding.detect.return_value = Result(err="not found")

        result = backend.search("bottle")
        assert result.ok
        assert result.result["status"] == "NOT_FOUND"

    def test_empty_index_returns_not_found(self, mock_backend):
        """An empty semantic index returns a clean NOT_FOUND, not an error."""
        backend, _tmp_path = mock_backend

        from utils.result import Result

        backend._mock_normalizer.normalize.return_value = Result(result="phone")
        backend._mock_semantic.embed_text.return_value = _unit_vector(768, seed=70)

        result = backend.search("phone")
        assert result.ok
        assert result.result["status"] == "NOT_FOUND"
        assert result.result["candidate_count"] == 0

    def test_grounding_false_positive_regression(self, mock_backend):
        """Image A has a Grounding DINO false positive; Image B has the object.

        The pipeline must return Image B, not Image A, because the
        combined semantic + grounding score for B is higher.
        This specifically protects against the old first-match bug.
        """
        backend, tmp_path = mock_backend

        # Image A: semantically distant from query, DINO false positive
        vec_a = _unit_vector(768, seed=80)
        # Image B: semantically close to query, DINO confirms
        vec_b = _unit_vector(768, seed=81)
        query_vec = vec_b.copy()  # query matches Image B

        self._ingest_image_with_embedding(
            backend, tmp_path, "img_a.jpg", vec_a, seed_vis=90
        )
        img_b_path = self._ingest_image_with_embedding(
            backend, tmp_path, "img_b.jpg", vec_b, seed_vis=91
        )

        from utils.result import Result

        backend._mock_normalizer.normalize.return_value = Result(result="glasses")
        backend._mock_semantic.embed_text.return_value = query_vec

        # Grounding returns true for both — this is the false-positive scenario.
        # Image A gets a high Grounding score despite being semantically distant.
        # The old pipeline would have returned the first image visited (oldest →
        # newest OR a false positive). The new pipeline must rank by combined score.
        backend._mock_grounding.detect.return_value = Result(
            result={
                "label": "glasses",
                "confidence": 0.85,
                "bounding_box": [0, 0, 16, 16],
            }
        )

        result = backend.search("glasses")
        assert result.ok
        data = result.result
        assert data["status"] == "FOUND"
        # Image B must win because its semantic_score (= 1.0) far outweighs A's
        assert data["image_path"] == img_b_path

    def test_object_absent_returns_not_found(self, mock_backend, monkeypatch):
        """When the query object is not in memory, NOT_FOUND is returned."""
        import config

        # Use a realistic threshold
        monkeypatch.setattr(config, "MATCH_THRESHOLD", 0.45)

        backend, tmp_path = mock_backend

        # Ingest an image of an unrelated object (a chair)
        vec_chair = _unit_vector(768, seed=100)
        self._ingest_image_with_embedding(
            backend, tmp_path, "chair.jpg", vec_chair, seed_vis=101
        )

        from utils.result import Result

        # Query for "glasses" — not in index; embedding is orthogonal to stored vec
        query_vec = _unit_vector(768, seed=102)
        backend._mock_normalizer.normalize.return_value = Result(result="glasses")
        backend._mock_semantic.embed_text.return_value = query_vec
        # Grounding also finds nothing
        backend._mock_grounding.detect.return_value = Result(err="not found")

        result = backend.search("glasses")
        assert result.ok
        assert result.result["status"] == "NOT_FOUND"


class TestDeletionPipeline:
    def test_vector_index_deletion(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ai.vector_index._INDEX_DIR", tmp_path)
        from ai.vector_index import VectorIndex

        idx = VectorIndex("test_del", 64)
        v1 = _unit_vector(64, seed=1)
        v2 = _unit_vector(64, seed=2)
        v3 = _unit_vector(64, seed=3)

        idx.add(v1, 10)
        idx.add(v2, 20)
        idx.add(v3, 30)

        assert idx.size == 3
        idx.remove_objects([10, 30])
        assert idx.size == 1

        # Searching for v2 should still yield 20
        res = idx.search(v2, k=1)
        assert res[0][0] == 20

        # Clear
        idx.clear()
        assert idx.size == 0
        assert idx.search(v2, k=1) == []

    def test_db_deletion(self, temp_db, tmp_path):
        img1 = tmp_path / "del1.jpg"
        img1.write_bytes(b"fake")
        img2 = tmp_path / "del2.jpg"
        img2.write_bytes(b"fake")

        r1 = temp_db.add_image(str(img1))
        r2 = temp_db.add_image(str(img2))

        o1 = temp_db.add_object_observation(
            int(r1.id), "a", 0.9, "[]", datetime.datetime.now(datetime.UTC)
        )
        temp_db.add_object_observation(
            int(r2.id), "b", 0.9, "[]", datetime.datetime.now(datetime.UTC)
        )

        assert temp_db.get_image_count() == 2

        deleted_obs = temp_db.delete_image(int(r1.id))
        assert deleted_obs == [o1]

        assert temp_db.get_image_count() == 1
        assert temp_db.get_image_by_id(int(r1.id)) is None
        assert temp_db.get_object_observations_by_ids([o1]) == []

        temp_db.clear_all()
        assert temp_db.get_image_count() == 0

    def test_backend_orchestration_delete_image(self, mock_backend):
        backend, tmp_path = mock_backend

        img_path = tmp_path / "backend_del.jpg"
        _write_fake_image(img_path)

        from ai.broad_detector import Detection

        backend._mock_broad.detect.return_value = [
            Detection(label="bag", confidence=0.9, bounding_box=[0, 0, 16, 16])
        ]
        backend._mock_semantic.embed_image.return_value = _unit_vector(768, seed=1)
        backend._mock_visual.embed.return_value = _unit_vector(384, seed=2)

        result = backend.ingest_image(str(img_path))
        assert result.ok
        assert backend._semantic_index.size == 1

        images = backend.get_all_images_desc()
        assert len(images) == 1

        # Now delete
        del_res = backend.delete_image(int(images[0].id))
        assert del_res.ok

        # Verify
        assert backend.get_image_count() == 0
        assert backend._semantic_index.size == 0
        assert not img_path.exists()

    def test_backend_orchestration_clear_all(self, mock_backend):
        backend, tmp_path = mock_backend

        # Ingest 2 images
        img_path1 = tmp_path / "backend_clr1.jpg"
        img_path2 = tmp_path / "backend_clr2.jpg"
        _write_fake_image(img_path1)
        _write_fake_image(img_path2)

        backend._mock_broad.detect.return_value = []
        backend.ingest_image(str(img_path1))
        backend.ingest_image(str(img_path2))

        assert backend.get_image_count() == 2

        # Call clear_all_memory
        res = backend.clear_all_memory()
        assert res.ok

        assert backend.get_image_count() == 0
