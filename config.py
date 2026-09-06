"""Pipeline configuration for Saarthi.

All tunable pipeline constants live here. Modify this file to adjust
retrieval behaviour without hunting through individual modules.
"""

# --- Retrieval ---

# Number of top object observations retrieved from the semantic index per query.
# Higher values increase recall at the cost of more expensive verification.
SEMANTIC_TOP_K: int = 20
GROUNDING_MAX_WORKERS: int = 4

# --- Candidate scoring weights ---
# Initial heuristic values only — calibrate with an evaluation dataset.
# Must sum to 1.0.

SEMANTIC_WEIGHT: float = 0.40  # SigLIP2 text-to-image similarity
VISUAL_WEIGHT: float = 0.25  # DINOv2 visual similarity
GROUNDING_WEIGHT: float = 0.35  # Grounding DINO Tiny detection confidence
RECENCY_WEIGHT: float = 0.15  # Recent observations are more likely to be intended

# Exponential decay half-life for observation timestamps.
RECENCY_HALF_LIFE_HOURS: float = 24.0

# --- Match decision ---

# Minimum final_score required to return a FOUND result.
# Candidates below this threshold produce a NOT_FOUND outcome.
# Set empirically from observed score distributions; tune as needed.
MATCH_THRESHOLD: float = 0.45

# Minimum score margin (best_score - second_best_score) required for FOUND.
# Set to 0.0 to disable the margin check (current default).
# Enabling this rejects near-tie results that are inherently ambiguous.
MIN_MARGIN: float = 0.0


# --- Score calibration (raw signal → [0, 1]) ---
# These map each raw signal onto a common [0, 1] scale before the weighted
# sum, because the three signals have incompatible raw distributions:
#   SigLIP2 cosine:      true match ≈ 0.10–0.28, unrelated ≈ 0.00–0.10
#   DINOv2 cosine:       same object ≈ 0.70–0.90, different ≈ 0.40–0.60
#   GD confidence:       true detection ≈ 0.35–0.90, none = 0.0
# PLACEHOLDER values — replace with numbers measured from the debug
# 'candidates' list in search results (see verification steps at the end).
SEMANTIC_FLOOR: float = 0.03
SEMANTIC_CEILING: float = 0.27
VISUAL_FLOOR: float = 0.45
VISUAL_CEILING: float = 0.80
GROUNDING_FLOOR: float = 0.25
GROUNDING_CEILING: float = 0.85

# Minimum IoU between Grounding DINO's verified box and the retrieved
# object's stored box required for the visual score to be counted.
VISUAL_IOU_GATE: float = 0.2
