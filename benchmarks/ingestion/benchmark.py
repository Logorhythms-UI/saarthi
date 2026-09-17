"""Time each stage of the ingestion pipeline on one photo.

Usage: uv run python benchmarks/ingestion/benchmark.py PHOTO [--runs 3] [--json OUT]

The first run of each stage is a warm-up and is not counted. The numbers
show which detector passes the hardware can afford (see ai/config.toml).
"""

import argparse
import json
import logging
import statistics
import time
from collections.abc import Callable
from pathlib import Path

from ai.api import AiEngine
from ai.settings import load_config
from ai.types import Candidate

logging.getLogger("ai").setLevel(logging.WARNING)


def measure(action: Callable[[], object], runs: int) -> tuple[float, object]:
    """Return the median time of `runs` calls in seconds, and the last result."""
    assert runs > 0, "runs must be positive"
    result: object = action()  # Warm-up
    times: list[float] = []
    for _ in range(runs):
        started: float = time.perf_counter()
        result = action()
        times.append(time.perf_counter() - started)
    return statistics.median(times), result


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--json", type=Path, default=None, help="write the results to this file"
    )
    args: argparse.Namespace = parser.parse_args()

    started: float = time.perf_counter()
    engine: AiEngine = AiEngine()
    load_seconds: float = time.perf_counter() - started
    data: bytes = args.photo.read_bytes()

    rows: list[tuple[str, float, str]] = []
    open_s, image = measure(lambda: engine.open_image(data), args.runs)
    rows.append(("open image", open_s, f"{image.width}x{image.height}"))
    quality_s, report = measure(lambda: engine.assess_quality(image), args.runs)
    blur_min: float = load_config().quality.blur_min
    verdict: str = "blurry" if engine.is_blurry(report) else "sharp enough"  # type: ignore[arg-type]
    rows.append(
        (
            "quality (blur + hash)",
            quality_s,
            f"blur score {report.blur_score:.0f} (minimum {blur_min:.0f}): {verdict}",  # type: ignore[attr-defined]
        )
    )

    candidates: list[Candidate] = []
    for name, options in [
        ("detect: full frame", {}),
        ("detect: full + tiling", {"tiling": True}),
        ("detect: full + generic", {"include_generic": True}),
        ("detect: all three passes", {"tiling": True, "include_generic": True}),
    ]:
        seconds, found = measure(
            lambda o=options: engine.detect_candidates(image, **o), args.runs
        )
        rows.append((name, seconds, f"{len(found)} candidates"))  # type: ignore[arg-type]
        if not options:
            candidates = found  # type: ignore[assignment]

    boxes = [c.box for c in candidates]
    embed_s, _ = measure(lambda: engine.embed_regions(image, boxes), args.runs)
    rows.append(("embed regions", embed_s, f"{len(boxes)} boxes"))

    print(f"\nModels loaded in {load_seconds:.1f} s. Median of {args.runs} run(s).\n")
    print(f"{'stage':<28}{'seconds':>9}  note")
    for stage, seconds, note in rows:
        print(f"{stage:<28}{seconds:>9.2f}  {note}")
    full: float = next(s for n, s, _ in rows if n == "detect: full frame")
    print(f"\nOne image with the default passes: about {full + embed_s:.1f} s")

    if args.json is not None:
        args.json.write_text(
            json.dumps({"load_seconds": load_seconds, "rows": rows}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
