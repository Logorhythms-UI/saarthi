"""Post-processing of detector output. Pure functions with no model.

Order of the steps (see `select_candidates`):
1. Map each label to a phrase of the vocabulary.
2. Keep the best boxes of each phrase.
3. Clip the boxes and drop those with a bad size.
4. Merge duplicate boxes (class-agnostic NMS) and keep the union of labels.
5. Drop objects that lie inside a screen, a picture frame or a mirror.
6. Flag objects that a hand covers.
"""

import dataclasses
import re
from collections.abc import Callable, Sequence

from PIL import Image

from ai.geometry import clip_box, iou, nms, overlap_fraction, touches_edge
from ai.quality import blur_score
from ai.settings import DetectionConfig
from ai.types import Box, Candidate, RawDetection, Source

# Maximum length of a text prompt for the detector, in tokens.
MAX_PROMPT_TOKENS: int = 256

_ARTICLES: frozenset[str] = frozenset({"a", "an", "the", "some"})
_FILLERS: frozenset[str] = frozenset({"of"})


def _words(text: str) -> list[str]:
    """Return the lowercase words of a text, without articles."""
    return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in _ARTICLES]


def _head(words: list[str]) -> str:
    """Return the head noun: the last word that is not a filler."""
    content: list[str] = [w for w in words if w not in _FILLERS]
    return content[-1] if content else ""


def match_phrase(label: str, vocabulary: Sequence[str]) -> str:
    """Return the phrase of the vocabulary that a detector label stands for.

    The detector can return a part of a phrase ("bottle" for "a water bottle").
    Match in this order: same words, same head noun, label words inside the
    phrase. If nothing matches, return the label as it is.
    """
    assert isinstance(label, str), "label must be a string"
    label_words: list[str] = _words(label)
    if not label_words:
        return label.strip()
    entries: list[tuple[str, list[str]]] = [(p, _words(p)) for p in vocabulary]

    for phrase, words in entries:
        if words == label_words:
            return phrase

    label_head: str = _head(label_words)
    same_head: list[tuple[str, list[str]]] = [
        (p, w) for p, w in entries if _head(w) == label_head
    ]
    if same_head:
        return max(same_head, key=lambda e: len(set(e[1]) & set(label_words)))[0]

    for phrase, words in entries:
        if set(label_words) <= set(words):
            return phrase
    return label.strip()


def plan_prompts(
    object_phrases: Sequence[str],
    extra_phrases: Sequence[str],
    chunk_size: int,
    count_tokens: Callable[[str], int],
) -> list[list[str]]:
    """Split the vocabulary into prompts for separate detector passes.

    Each prompt holds the extra phrases (hand, depictions) and as many object
    phrases as `chunk_size` and the token limit allow.
    """
    assert object_phrases, "object_phrases must not be empty"
    assert chunk_size > len(extra_phrases), "chunk_size must leave room for objects"

    def tokens(phrases: list[str]) -> int:
        # Two special tokens, and one period after each phrase.
        return 2 + sum(count_tokens(p) + 1 for p in phrases)

    if tokens(list(extra_phrases)) >= MAX_PROMPT_TOKENS:
        raise ValueError("The extra phrases alone fill the prompt")

    prompts: list[list[str]] = []
    current: list[str] = []
    for phrase in object_phrases:
        trial: list[str] = [*current, phrase]
        full: bool = len(trial) + len(extra_phrases) > chunk_size or (
            tokens([*trial, *extra_phrases]) > MAX_PROMPT_TOKENS
        )
        if full and current:
            prompts.append([*current, *extra_phrases])
            trial = [phrase]
        current = trial
    prompts.append([*current, *extra_phrases])
    return prompts


@dataclasses.dataclass(frozen=True)
class _Item:
    box: Box
    score: float
    source: Source
    phrase: str


def _cap_per_phrase(
    labelled: list[tuple[RawDetection, str]], limit: int
) -> list[tuple[RawDetection, str]]:
    """Keep the `limit` best boxes of each phrase."""
    counts: dict[str, int] = {}
    kept: list[tuple[RawDetection, str]] = []
    for detection, phrase in sorted(labelled, key=lambda e: -e[0].score):
        counts[phrase] = counts.get(phrase, 0) + 1
        if counts[phrase] <= limit:
            kept.append((detection, phrase))
    return kept


def select_candidates(
    detections: list[RawDetection],
    width: int,
    height: int,
    vocabulary: Sequence[str],
    config: DetectionConfig,
) -> list[Candidate]:
    """Turn raw detections into candidates. The best score comes first.

    `quality_ok` is True here. `apply_crop_quality` sets it later.
    """
    assert width > 0 and height > 0, "image size must be positive"
    known: list[str] = [
        *vocabulary,
        config.hand_phrase,
        *config.depiction_phrases,
        config.generic_phrase,
    ]
    depiction_phrases: frozenset[str] = frozenset(config.depiction_phrases)
    labelled: list[tuple[RawDetection, str]] = [
        (d, match_phrase(d.label, known)) for d in detections
    ]

    hands: list[Box] = []
    depictions: list[Box] = []
    objects: list[_Item] = []
    for detection, phrase in _cap_per_phrase(labelled, config.max_per_phrase):
        box: Box | None = clip_box(detection.box, width, height)
        if box is None or min(box.width, box.height) < config.min_box_side:
            continue
        if phrase == config.hand_phrase:
            hands.append(box)
        elif phrase in depiction_phrases:
            depictions.append(box)
        elif box.area > config.max_frame_fraction * width * height:
            continue
        else:
            objects.append(_Item(box, detection.score, detection.source, phrase))

    kept: list[int] = nms(
        [o.box for o in objects], [o.score for o in objects], config.nms_iou
    )
    candidates: list[Candidate] = []
    for index in kept:
        item: _Item = objects[index]
        if any(
            overlap_fraction(item.box, d) >= config.depiction_inside for d in depictions
        ):
            continue
        phrases: list[str] = []
        for other in objects:
            merged: bool = iou(item.box, other.box) >= config.nms_iou
            if merged and other.phrase and other.phrase not in phrases:
                phrases.append(other.phrase)
        in_hand: bool = any(
            overlap_fraction(item.box, h) > config.hand_overlap for h in hands
        )
        candidates.append(
            Candidate(item.box, item.score, item.source, tuple(phrases), in_hand, True)
        )
    return candidates


def apply_crop_quality(
    candidates: list[Candidate],
    image: Image.Image,
    min_crop_side: int,
    crop_blur_min: float,
) -> list[Candidate]:
    """Set `quality_ok`. A candidate is good if its crop is big enough, sharp
    enough, and not cut by the image edge.
    """
    assert isinstance(image, Image.Image), "image must be a PIL image"
    width, height = image.size
    result: list[Candidate] = []
    for candidate in candidates:
        box: Box = candidate.box
        good: bool = (
            min(box.width, box.height) >= min_crop_side
            and not touches_edge(box, width, height)
            and blur_score(image.crop((box.x0, box.y0, box.x1, box.y1)))
            >= crop_blur_min
        )
        result.append(dataclasses.replace(candidate, quality_ok=good))
    return result
