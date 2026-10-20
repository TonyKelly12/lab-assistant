"""Breadboard hole geometry: holes land on the board, spaced by pitch, at any angle."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labassist import breadboard as bb  # noqa: E402


def make_board(center, short, long_, angle_deg):
    """A board dict shaped like labassist.board.find_boards output (bench mm)."""
    a = math.radians(angle_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    hl, hs = long_ / 2, short / 2
    local = np.array([[-hl, -hs], [hl, -hs], [hl, hs], [-hl, hs]])  # long along local x
    corners = local @ R.T + np.asarray(center, float)
    return {"center": list(center), "size": [short, long_], "angle": angle_deg % 180,
            "corners": corners.round(2).tolist()}


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def test_profile_autopick():
    assert bb.pick_profile(make_board([600, 300], 55, 165, 0)).name == "full"
    assert bb.pick_profile(make_board([600, 300], 46, 83, 0)).name == "half"


def test_adjacent_columns_one_pitch_apart():
    board = make_board([600, 300], 55, 165, 0)
    p = bb.pick_profile(board)
    c1 = bb.hole_xy(board, p, 1, p.rows["a"])
    c2 = bb.hole_xy(board, p, 2, p.rows["a"])
    assert abs(_dist(c1, c2) - bb.PITCH) < 0.05, (c1, c2)


def test_adjacent_rows_spacing():
    board = make_board([600, 300], 55, 165, 0)
    p = bb.pick_profile(board)
    a = bb.hole_xy(board, p, 5, p.rows["a"])
    b = bb.hole_xy(board, p, 5, p.rows["b"])
    assert abs(_dist(a, b) - (p.rows["b"] - p.rows["a"])) < 0.05


def test_axes_are_perpendicular_when_rotated():
    board = make_board([600, 300], 55, 165, 30)
    p = bb.pick_profile(board)
    # wide baselines so 0.1 mm hole rounding doesn't dominate the angle
    col_dir = np.subtract(bb.hole_xy(board, p, 21, p.rows["a"]), bb.hole_xy(board, p, 1, p.rows["a"]))
    row_dir = np.subtract(bb.hole_xy(board, p, 1, p.rows["j"]), bb.hole_xy(board, p, 1, p.rows["a"]))
    cosang = col_dir @ row_dir / (np.linalg.norm(col_dir) * np.linalg.norm(row_dir))
    assert abs(cosang) < 0.02, cosang


def test_holes_stay_inside_the_board():
    board = make_board([600, 300], 55, 165, 17)
    p = bb.pick_profile(board)
    corners = np.asarray(board["corners"])
    # distance from center to each hole must be within the board's half-diagonal
    half_diag = 0.5 * math.hypot(55, 165) + 0.5
    for c in (1, 20, 63):
        for r in ("a", "e", "f", "j"):
            h = bb.hole_xy(board, p, c, p.rows[r])
            assert _dist(h, board["center"]) <= half_diag, (c, r, h)
    assert corners.shape == (4, 2)


def test_resolve_column_row_hole():
    board = make_board([600, 300], 55, 165, 0)
    col = bb.resolve("light up column 12", board)
    assert col["kind"] == "column" and col["label"] == "column 12" and len(col["holes"]) == 10

    row = bb.resolve("show row e", board)
    assert row["kind"] == "row" and len(row["holes"]) == 63

    hole = bb.resolve("hole e12", board)
    assert hole["kind"] == "hole" and len(hole["holes"]) == 1

    rails = bb.resolve("highlight the power rails", board)
    assert rails["kind"] == "rail" and len(rails["holes"]) > 0

    place = bb.resolve("place the ESP32 at column 10", board)
    assert place["kind"] == "place" and place.get("footprint") and len(place["footprint"]) == 4

    assert bb.resolve("do a backflip", board)["kind"] == "unknown"


def test_scene_has_a_dot_per_hole_plus_label():
    board = make_board([600, 300], 55, 165, 0)
    r = bb.resolve("column 12", board)
    scene = bb.holes_scene(r)
    dots = [i for i in scene["items"] if i["type"] == "circle"]
    texts = [i for i in scene["items"] if i["type"] == "text"]
    assert len(dots) == 10 and len(texts) == 1


def test_out_of_range_column_is_rejected():
    board = make_board([600, 300], 55, 165, 0)
    assert bb.resolve("column 99", board)["kind"] == "unknown"
