"""Calibration maths, kept free of network code so it can be tested offline.

Frames:
  bench      millimetres on the bench; origin at printed marker 0, x to the
             right, y toward you (front of the bench)
  camera     pixels in the overhead camera's full-resolution snapshot
  projector  pixels in the projector's output (e.g. 1920x1080)

Result: H_bench_to_proj = H_cam_to_proj @ H_bench_to_cam
"""
import cv2
import numpy as np

from . import aruco
from .geometry import fit_homography, reprojection_error, undistort_points


def projector_pattern(width, height, cols=6, rows=4, side=None):
    """Grid of ArUco markers to project during calibration.

    Returns (image uint8 HxW, layout) where layout maps marker id -> (4, 2)
    corner positions in projector pixels.
    """
    if side is None:
        side = int(min(width / (cols * 1.6), height / (rows * 1.6)))
    img = np.full((height, width), 255, np.uint8)
    layout = {}
    xs = np.linspace(side * 0.75, width - side * 1.75, cols)
    ys = np.linspace(side * 0.75, height - side * 1.75, rows)
    marker_id = aruco.PROJECTED_MARKER_IDS.start
    for y in ys:
        for x in xs:
            x0, y0 = int(round(x)), int(round(y))
            img[y0:y0 + side, x0:x0 + side] = aruco.marker_image(marker_id, side)
            layout[marker_id] = np.array(
                [[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]], np.float32)
            marker_id += 1
    return img, layout


def focus_pattern(width, height):
    """Full-screen focus chart in projector pixels (white on black).

    A coarse grid over the whole image, plus a focus target (Siemens star,
    1-px line patches, small text) in the centre and each corner, so you can
    check sharpness where you work and at the edges. Turn the focus ring until
    the star's centre and the 1-px lines are crisp.
    """
    img = np.zeros((height, width), np.uint8)
    for x in range(0, width, 80):
        img[:, x] = 70
    for y in range(0, height, 80):
        img[y, :] = 70
    img[:, width - 1] = 70
    img[height - 1, :] = 70

    def target(cx, cy, r):
        # Siemens star: 36 alternating wedges; blur shows as a grey blob in the middle.
        for i in range(0, 36, 2):
            a0, a1 = np.deg2rad(i * 10), np.deg2rad((i + 1) * 10)
            pts = np.array([[cx, cy],
                            [cx + r * np.cos(a0), cy + r * np.sin(a0)],
                            [cx + r * np.cos(a1), cy + r * np.sin(a1)]], np.int32)
            cv2.fillConvexPoly(img, pts, 255)
        cv2.circle(img, (cx, cy), r, 255, 1)
        # 1-px vertical and horizontal line patches beside the star.
        x0, y0 = cx + r + 12, cy - r
        img[y0:y0 + 2 * r, x0:x0 + r] = 0
        img[y0:y0 + 2 * r, x0:x0 + r][:, ::2] = 255                  # vertical lines
        img[y0:y0 + 2 * r, x0 + r + 8:x0 + 2 * r + 8] = 0
        img[y0:y0 + 2 * r, x0 + r + 8:x0 + 2 * r + 8][::2, :] = 255  # horizontal lines
        # Small text at a few sizes.
        for j, scale in enumerate((0.35, 0.5, 0.7)):
            cv2.putText(img, "Pin 1 VCC GND 2.54", (cx - r, cy + r + 18 + j * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, 255, 1, cv2.LINE_AA)

    r = max(24, min(width, height) // 18)
    m = r + 24
    for cx, cy in [(width // 2, height // 2), (m, m), (width - 3 * r - m, m),
                   (m, height - r - m - 60), (width - 3 * r - m, height - r - m - 60)]:
        target(cx, cy, r)
    cv2.putText(img, "FOCUS CHART - make the centre target sharp, then check the corners",
                (width // 2 - 380, height // 2 - r - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 1, cv2.LINE_AA)
    return img


def solve_cam_to_proj(cam_image, layout, intrinsics=None, min_markers=4):
    """Find H mapping camera pixels -> projector pixels from a snapshot of the projected pattern."""
    found = aruco.detect(cam_image)
    cam_pts, proj_pts = [], []
    for marker_id, proj_corners in layout.items():
        if marker_id in found:
            cam_pts.append(found[marker_id])
            proj_pts.append(proj_corners)
    if len(cam_pts) < min_markers:
        raise ValueError(
            f"only {len(cam_pts)} of {len(layout)} projected markers seen (need {min_markers}). "
            "Darken the room, clear the bench, check focus, and make sure the camera sees the projection.")
    cam_pts = undistort_points(np.concatenate(cam_pts), intrinsics)
    proj_pts = np.concatenate(proj_pts)
    H, inliers = fit_homography(cam_pts, proj_pts, ransac=True)
    mean_err, max_err = reprojection_error(H, cam_pts[inliers], proj_pts[inliers])
    return H, {
        "markers_seen": len(cam_pts) // 4,
        "markers_total": len(layout),
        "inlier_points": int(inliers.sum()),
        "points": len(inliers),
        "mean_error_px": round(mean_err, 3),
        "max_error_px": round(max_err, 3),
    }


def solve_bench_to_cam(cam_image, bench_markers, intrinsics=None):
    """Find H mapping bench mm -> camera pixels from printed markers at known positions.

    bench_markers: {id: (x_mm, y_mm)} positions of each printed marker's centre.
    """
    found = aruco.detect(cam_image)
    missing = [m for m in bench_markers if m not in found]
    if len(bench_markers) - len(missing) < 4:
        raise ValueError(
            f"bench markers missing from the camera view: {missing}. "
            "All four printed markers must be visible (and not covered by the projection pattern).")
    ids = [m for m in bench_markers if m in found]
    cam_pts = undistort_points(np.array([aruco.center(found[m]) for m in ids]), intrinsics)
    bench_pts = np.array([bench_markers[m] for m in ids], np.float64)
    H, _ = fit_homography(bench_pts, cam_pts, ransac=False)
    mean_err, max_err = reprojection_error(H, bench_pts, cam_pts)
    return H, {"markers_seen": ids, "mean_error_px": round(mean_err, 3), "max_error_px": round(max_err, 3)}


def bench_area(bench_markers, margin_mm=0.0):
    """Drawable area [x0, y0, x1, y1] in bench mm: the markers' bounding box plus a margin."""
    pts = np.array(list(bench_markers.values()), np.float64)
    m = float(margin_mm)
    return [float(pts[:, 0].min()) - m, float(pts[:, 1].min()) - m,
            float(pts[:, 0].max()) + m, float(pts[:, 1].max()) + m]
