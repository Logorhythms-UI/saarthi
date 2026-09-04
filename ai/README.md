# AI

This module owns all AI/ML model loading and inference for Saarthi.

Other modules should use the interfaces defined here rather than
directly instantiating or manipulating model internals.

## Components

| File | Class | Model | Purpose |
|------|-------|-------|---------|
| `broad_detector.py` | `BroadDetector` | YOLOv8n | High-recall object proposals during ingestion |
| `semantic_embedder.py` | `SemanticEmbedder` | SigLIP2-base | Joint image + text embeddings for retrieval |
| `visual_embedder.py` | `VisualEmbedder` | DINOv2-small | Visual similarity embeddings for secondary ranking |
| `object_detector.py` | `ObjectDetector` | Grounding DINO Tiny | Open-vocabulary verification during search |
| `query_normalizer.py` | `QueryNormalizer` | Llama 3.2 1B (GGUF) | Natural-language query → visual search phrase |
| `prompts.py` | — | — | System prompt constants |

## Ingestion flow

```
Uploaded image
    ↓
BroadDetector.detect()       ← high-recall, all objects
    ↓
object crops (in memory)
    ↓
SemanticEmbedder.embed_image()
VisualEmbedder.embed()
    ↓
embeddings → VectorIndex (FAISS)
```

## Search flow

```
User query
    ↓
QueryNormalizer.normalize()
    ↓
SemanticEmbedder.embed_text()
    ↓
VectorIndex.search()          ← top-K object candidates
    ↓
ObjectDetector.detect()       ← verification on candidate images
    ↓
scored & ranked result
```

## Model lifecycle

All models are loaded once at application startup (inside `Backend.__init__`)
and remain in memory for the application lifetime.
Do not instantiate model classes inside request handlers or search loops.
