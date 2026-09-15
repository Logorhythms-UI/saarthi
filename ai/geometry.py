"""Box geometry. Pure functions with no model and no image."""

from dataclasses import dataclass

from ai.types import Box


def intersection_area(a: Box, b: Box) -> int:
    """Return the area where two boxes overlap."""
    width: int = min(a.x1, b.x1) - max(a.x0, b.x0)
    height: int = min(a.y1, b.y1) - max(a.y0, b.y0)
    return width * height if width > 0 and height > 0 else 0


def iou(a: Box, b: Box) -> float:
    """Return intersection over union of two boxes."""
    inter: int = intersection_area(a, b)
    union: int = a.area + b.area - inter
    return inter / union


def overlap_fraction(inner: Box, outer: Box) -> float:
    """Return the part of `inner` that lies inside `outer`, from 0.0 to 1.0."""
    return intersection_area(inner, outer) / inner.area


def clip_box(box: Box, width: int, height: int) -> Box | None:
    """Cut a box to the image. Return None if nothing is left."""
    assert width > 0 and height > 0, "image size must be positive"
    x0: int = max(box.x0, 0)
    y0: int = max(box.y0, 0)
    x1: int = min(box.x1, width)
    y1: int = min(box.y1, height)
    if x0 >= x1 or y0 >= y1:
        return None
    return Box(x0, y0, x1, y1)


def touches_edge(box: Box, width: int, height: int) -> bool:
    """Return True if the box reaches the border of the image."""
    return box.x0 <= 0 or box.y0 <= 0 or box.x1 >= width or box.y1 >= height


def nms(boxes: list[Box], scores: list[float], iou_threshold: float) -> list[int]:
    """Class-agnostic non-maximum suppression.

    Return the indices of the boxes to keep, best score first. A box is
    dropped when its IoU with a kept box is at or above the threshold.
    """
    assert len(boxes) == len(scores), "boxes and scores must have the same length"
    assert 0.0 < iou_threshold <= 1.0, "iou_threshold must be in (0, 1]"
    order: list[int] = sorted(range(len(boxes)), key=lambda i: -scores[i])
    kept: list[int] = []
    for index in order:
        if all(iou(boxes[index], boxes[k]) < iou_threshold for k in kept):
            kept.append(index)
    return kept


@dataclass(frozen=True)
class Tile:
    """The position of a tile in the full image."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def box(self) -> Box:
        return Box(self.x0, self.y0, self.x1, self.y1)


def tile_grid(width: int, height: int, overlap: float) -> list[Tile]:
    """Split an image into 2 x 2 tiles. `overlap` is the part of a tile that two neighbours share."""
    assert width >= 2 and height >= 2, "image is too small to tile"
    assert 0.0 <= overlap < 1.0, "overlap must be in [0, 1)"
    tile_width: int = min(width, round(width / (2.0 - overlap)))
    tile_height: int = min(height, round(height / (2.0 - overlap)))
    xs: tuple[int, int] = (0, width - tile_width)
    ys: tuple[int, int] = (0, height - tile_height)
    return [Tile(x, y, x + tile_width, y + tile_height) for y in ys for x in xs]


def map_tile_box(box: Box, tile: Tile) -> Box:
    """Move a box from tile coordinates to image coordinates."""
    return Box(box.x0 + tile.x0, box.y0 + tile.y0, box.x1 + tile.x0, box.y1 + tile.y0)
