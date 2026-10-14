"""Offline tests: synthetic camera images with known geometry, no hardware needed.

Run:  python3 -m pytest tests/   (or: python3 tests/test_calibration.py)
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labassist import aruco, calib  # noqa: E402
from labassist.geometry import apply, css_matrix3d  # noqa: E402

CAM_W, CAM_H = 2592, 1944
PROJ_W, PROJ_H = 1920, 1080
BENCH = {0: (0.0, 0.0), 1: (900.0, 0.0), 2: (900.0, 450.0), 3: (0.0, 450.0)}

# Ground truth: where the bench and the projector land in the camera image.
# Slightly tilted views, like a camera beside a projector on a tripod.
BENCH_CORNERS_IN_CAM = np.float32([[300, 380], [2330, 330], [2420, 1560], [220, 1640]])
PROJ_CORNERS_IN_CAM = np.float32([[160, 240], [2450, 200], [2520, 1720], [120, 1780]])
MARGIN_MM = 60  # bench area drawn around the marker rectangle


def truth():
    bench_rect = np.float32([[-MARGIN_MM, -MARGIN_MM], [900 + MARGIN_MM, -MARGIN_MM],
                             [900 + MARGIN_MM, 450 + MARGIN_MM], [-MARGIN_MM, 450 + MARGIN_MM]])
    H_bench_to_cam = cv2.getPerspectiveTransform(bench_rect, BENCH_CORNERS_IN_CAM)
    proj_rect = np.float32([[0, 0], [PROJ_W, 0], [PROJ_W, PROJ_H], [0, PROJ_H]])
    H_proj_to_cam = cv2.getPerspectiveTransform(proj_rect, PROJ_CORNERS_IN_CAM)
    return H_bench_to_cam, H_proj_to_cam


def bench_image(H_bench_to_cam, scale=4):
    """Camera view of the bench with the four printed 60 mm markers, projector dark."""
    w_mm, h_mm = 900 + 2 * MARGIN_MM, 450 + 2 * MARGIN_MM
    canvas = np.full((h_mm * scale, w_mm * scale), 170, np.uint8)
    side = 60 * scale
    for marker_id, (x, y) in BENCH.items():
        cx, cy = int((x + MARGIN_MM) * scale), int((y + MARGIN_MM) * scale)
        pad = 8 * scale
        canvas[cy - side // 2 - pad:cy + side // 2 + pad, cx - side // 2 - pad:cx + side // 2 + pad] = 255
        canvas[cy - side // 2:cy - side // 2 + side, cx - side // 2:cx - side // 2 + side] = aruco.marker_image(marker_id, side)
    to_canvas = np.array([[scale, 0, MARGIN_MM * scale], [0, scale, MARGIN_MM * scale], [0, 0, 1]], np.float64)
    H = H_bench_to_cam @ np.linalg.inv(to_canvas)
    img = cv2.warpPerspective(canvas, H, (CAM_W, CAM_H), borderValue=40)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def pattern_image(H_proj_to_cam, layout_img):
    lit = cv2.warpPerspective(layout_img, H_proj_to_cam, (CAM_W, CAM_H), borderValue=0)
    img = (40 + lit.astype(np.float32) * 0.75).clip(0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)  # a little projector blur
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_end_to_end():
    H_bench_to_cam_true, H_proj_to_cam_true = truth()
    H_bench_to_proj_true = np.linalg.inv(H_proj_to_cam_true) @ H_bench_to_cam_true

    H_bench_to_cam, rep_b = calib.solve_bench_to_cam(bench_image(H_bench_to_cam_true), BENCH)
    pattern, layout = calib.projector_pattern(PROJ_W, PROJ_H)
    H_cam_to_proj, rep_p = calib.solve_cam_to_proj(pattern_image(H_proj_to_cam_true, pattern), layout)
    H = H_cam_to_proj @ H_bench_to_cam

    probe = np.array([[x, y] for x in np.linspace(0, 900, 7) for y in np.linspace(0, 450, 5)])
    err = np.linalg.norm(apply(H, probe) - apply(H_bench_to_proj_true, probe), axis=1)
    print("bench report", rep_b)
    print("projector report", rep_p)
    print(f"bench->projector error: mean {err.mean():.2f} px, max {err.max():.2f} px")
    assert rep_p["markers_seen"] == rep_p["markers_total"]
    assert err.max() < 2.0, err.max()


def test_reduced_scale_detection_keeps_accuracy():
    """OpenCV 4.6 (Pi OS Bookworm) only finds markers in big frames at reduced
    scale. Force that path and check calibration stays accurate."""
    original = aruco.detect
    try:
        for scales in ((0.5,), (0.25,)):
            calib.aruco.detect = lambda img, s=scales: original(img, scales=s)
            H_bench_to_cam_true, H_proj_to_cam_true = truth()
            H_true = np.linalg.inv(H_proj_to_cam_true) @ H_bench_to_cam_true
            H_bc, _ = calib.solve_bench_to_cam(bench_image(H_bench_to_cam_true), BENCH)
            pattern, layout = calib.projector_pattern(PROJ_W, PROJ_H)
            H_cp, rep = calib.solve_cam_to_proj(pattern_image(H_proj_to_cam_true, pattern), layout)
            probe = np.array([[x, y] for x in np.linspace(0, 900, 7) for y in np.linspace(0, 450, 5)])
            err = np.linalg.norm(apply(H_cp @ H_bc, probe) - apply(H_true, probe), axis=1)
            print(f"detection at scale {scales[0]}: bench->projector error mean {err.mean():.2f} px, max {err.max():.2f} px")
            assert rep["markers_seen"] == rep["markers_total"]
            assert err.max() < 2.0, (scales, err.max())
    finally:
        calib.aruco.detect = original


def test_css_matrix3d_matches_homography():
    H = np.array([[1.2, 0.1, 30], [0.05, 0.9, 12], [0.0004, 0.0002, 1.0]])
    vals = [float(v) for v in css_matrix3d(H)[len("matrix3d("):-1].split(",")]
    M = np.array(vals).reshape(4, 4).T  # column-major -> row-major
    for x, y in [(0, 0), (100, 50), (900, 450)]:
        v = M @ np.array([x, y, 0, 1])
        expected = apply(H, [[x, y]])[0]
        assert np.allclose(v[:2] / v[3], expected), (v, expected)


if __name__ == "__main__":
    test_end_to_end()
    test_reduced_scale_detection_keeps_accuracy()
    test_css_matrix3d_matches_homography()
    print("all tests passed")
