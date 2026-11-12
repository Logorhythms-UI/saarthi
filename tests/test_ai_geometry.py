import pytest

from ai.geometry import (
    Tile,
    clip_box,
    intersection_area,
    iou,
    map_tile_box,
    nms,
    overlap_fraction,
    tile_grid,
    touches_edge,
)
from ai.types import Box


def test_box_properties_and_validation() -> None:
    box: Box = Box(10, 20, 110, 70)
    assert (box.width, box.height, box.area) == (100, 50, 5000)
    with pytest.raises(AssertionError):
        Box(5, 5, 5, 10)


def test_iou_and_overlap() -> None:
    a: Box = Box(0, 0, 10, 10)
    assert iou(a, a) == 1.0
    assert iou(a, Box(20, 20, 30, 30)) == 0.0
    assert iou(a, Box(5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert intersection_area(a, Box(10, 0, 20, 10)) == 0  # Touching is no overlap
    assert overlap_fraction(Box(0, 0, 10, 10), Box(0, 0, 5, 10)) == 0.5
    assert overlap_fraction(Box(2, 2, 4, 4), Box(0, 0, 10, 10)) == 1.0


def test_clip_box() -> None:
    assert clip_box(Box(-5, -5, 50, 50), 40, 30) == Box(0, 0, 40, 30)
    assert clip_box(Box(100, 100, 120, 120), 40, 30) is None


def test_touches_edge() -> None:
    assert touches_edge(Box(0, 5, 10, 10), 100, 100)
    assert touches_edge(Box(5, 5, 100, 10), 100, 100)
    assert not touches_edge(Box(1, 1, 99, 99), 100, 100)


def test_nms_keeps_the_best_of_overlapping_boxes() -> None:
    boxes: list[Box] = [Box(0, 0, 10, 10), Box(1, 1, 11, 11), Box(50, 50, 60, 60)]
    # The first two boxes have an IoU of about 0.68.
    assert nms(boxes, [0.5, 0.9, 0.4], 0.5) == [1, 2]
    assert nms(boxes, [0.5, 0.9, 0.4], 0.7) == [1, 0, 2]
    assert nms([], [], 0.7) == []


def test_tile_grid_covers_the_image_with_the_overlap() -> None:
    tiles: list[Tile] = tile_grid(1000, 800, 0.2)
    assert len(tiles) == 4
    assert {(t.x0, t.y0) for t in tiles} == {
        (0, 0),
        (tiles[1].x0, 0),
        (0, tiles[2].y0),
        (tiles[1].x0, tiles[2].y0),
    }
    assert max(t.x1 for t in tiles) == 1000 and max(t.y1 for t in tiles) == 800
    width: int = tiles[0].x1 - tiles[0].x0
    shared: int = tiles[0].x1 - tiles[1].x0
    assert shared / width == pytest.approx(0.2, abs=0.01)


def test_tile_grid_without_overlap_has_no_gap() -> None:
    tiles: list[Tile] = tile_grid(100, 100, 0.0)
    assert tiles[0].x1 == tiles[1].x0 == 50


def test_map_tile_box_adds_the_tile_offset() -> None:
    tile: Tile = Tile(500, 400, 1000, 800)
    assert map_tile_box(Box(10, 20, 30, 40), tile) == Box(510, 420, 530, 440)
