"""Board finder: a white rectangle drawn at a known bench position is found there."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labassist import board  # noqa: E402
from labassist.geometry import apply  # noqa: E402


def test_finds_breadboard_at_known_position():
    # Tilted camera: bench mm -> camera px.
    src = np.float32([[0, 0], [676, 0], [676, 378], [0, 378]])
    dst = np.float32([[500, 700], [1900, 690], [2050, 1500], [380, 1520]])
    H_bench_to_cam = cv2.getPerspectiveTransform(src, dst)
    H_cam_to_bench = np.linalg.inv(H_bench_to_cam)

    img = np.full((1944, 2592), 40, np.uint8)
    noise = np.random.default_rng(1).normal(0, 4, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    # A 55 x 83 mm board centred at (390, 200), turned 3 degrees.
    c, a = np.array([390.0, 200.0]), np.radians(3)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    corners = np.array([[-27.5, -41.5], [27.5, -41.5], [27.5, 41.5], [-27.5, 41.5]]) @ R.T + c
    cv2.fillConvexPoly(img, apply(H_bench_to_cam, corners).astype(np.int32), 200)
    # A bright marker plate that must be ignored.
    cv2.fillConvexPoly(img, apply(H_bench_to_cam, [[-40, -40], [40, -40], [40, 40], [-40, 40]]).astype(np.int32), 210)

    mask = board.bench_mask(img.shape, H_cam_to_bench, [-60, -60, 736, 438], exclude_mm=[(0, 0)])
    found, _ = board.find_boards(img, H_cam_to_bench, mask)
    assert len(found) == 1, found
    b = found[0]
    assert abs(b["center"][0] - 390) < 1.5 and abs(b["center"][1] - 200) < 1.5, b["center"]
    assert abs(b["size"][0] - 55) < 2 and abs(b["size"][1] - 83) < 2, b["size"]
    assert abs(b["angle"] - 93) < 1.5, b["angle"]
    assert len(board.outline_scene(found)["items"]) == 3
