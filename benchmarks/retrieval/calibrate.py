"""Calibrate retrieval score ranges from a labeled JSONL dataset.

Each input line must contain raw candidate scores and a binary label::

    {"semantic_score": 0.18, "visual_score": 0.77,
     "grounding_score": 0.82, "matched": true}

The tool writes 5th/95th percentile calibration ranges and a precision/recall
curve for the recorded final scores. It intentionally does not change config.py
because calibration data must be reviewed before production thresholds change.
"""

import argparse
import json
from pathlib import Path

import numpy as np

_SCORE_NAMES: tuple[str, ...] = (
    "semantic_score",
    "visual_score",
    "grounding_score",
)


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
        if not all(name in row for name in (*_SCORE_NAMES, "matched")):
            raise ValueError(
                f"Line {line_number} must contain {_SCORE_NAMES} and matched"
            )
        rows.append(row)
    if not rows:
        raise ValueError("Evaluation dataset is empty")
    if not any(bool(row["matched"]) for row in rows):
        raise ValueError("Evaluation dataset needs at least one positive match")
    if all(bool(row["matched"]) for row in rows):
        raise ValueError("Evaluation dataset needs at least one negative match")
    return rows


def _percentile_ranges(rows: list[dict]) -> dict[str, dict[str, float]]:
    ranges: dict[str, dict[str, float]] = {}
    for name in _SCORE_NAMES:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        ranges[name] = {
            "floor": float(np.percentile(values, 5)),
            "ceiling": float(np.percentile(values, 95)),
        }
    return ranges


def _precision_recall_curve(rows: list[dict]) -> list[dict[str, float]]:
    scores = np.asarray(
        [float(row.get("final_score", row["semantic_score"])) for row in rows]
    )
    labels = np.asarray([bool(row["matched"]) for row in rows])
    curve: list[dict[str, float]] = []
    for threshold in sorted(set(scores), reverse=True):
        predicted = scores >= threshold
        true_positive = int(np.sum(predicted & labels))
        false_positive = int(np.sum(predicted & ~labels))
        false_negative = int(np.sum(~predicted & labels))
        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive)
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative)
            else 0.0
        )
        curve.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
            }
        )
    return curve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Labeled JSONL evaluation dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("calibration.json"),
        help="Output JSON report path",
    )
    args = parser.parse_args()

    rows = _load_rows(args.input)
    report = {
        "sample_count": len(rows),
        "ranges": _percentile_ranges(rows),
        "precision_recall": _precision_recall_curve(rows),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
