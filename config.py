"""Pipeline configuration for Saarthi.

All tunable pipeline constants live here. Modify this file to adjust
retrieval behaviour without hunting through individual modules.
"""

# --- Retrieval ---

# Number of top object observations retrieved from the semantic index per query.
# Higher values increase recall at the cost of more expensive verification.
SEMANTIC_TOP_K: int = 20

# --- Candidate scoring weights ---
# Initial heuristic values only — calibrate with an evaluation dataset.
# Must sum to 1.0.

SEMANTIC_WEIGHT: float = 0.40  # SigLIP2 text-to-image similarity
VISUAL_WEIGHT: float = 0.25  # DINOv2 visual similarity
GROUNDING_WEIGHT: float = 0.35  # Grounding DINO Tiny detection confidence

# --- Match decision ---

# Minimum final_score required to return a FOUND result.
# Candidates below this threshold produce a NOT_FOUND outcome.
# Set empirically from observed score distributions; tune as needed.
MATCH_THRESHOLD: float = 0.45

# Minimum score margin (best_score - second_best_score) required for FOUND.
# Set to 0.0 to disable the margin check (current default).
# Enabling this rejects near-tie results that are inherently ambiguous.
MIN_MARGIN: float = 0.0
