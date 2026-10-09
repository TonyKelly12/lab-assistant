"""Homography helpers shared by calibration and rendering."""
import json

import cv2
import numpy as np


def undistort_points(points, intrinsics):
    """Remove lens distortion from pixel points.

    points: (N, 2) array in camera pixels. intrinsics: dict with K, D
    (from lens_calibrate.py) or None to pass points through unchanged.
    Returned points stay in pixel units (re-projected with K).
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if not intrinsics:
        return pts.reshape(-1, 2)
    K = np.array(intrinsics["K"], dtype=np.float64)
    D = np.array(intrinsics["D"], dtype=np.float64)
    return cv2.undistortPoints(pts, K, D, P=K).reshape(-1, 2)


def scale_intrinsics(intrinsics, width, height):
    """Rescale intrinsics calibrated at one resolution to another of the same aspect."""
    if not intrinsics:
        return None
    w0, h0 = intrinsics["image_size"]
    sx, sy = width / w0, height / h0
    K = np.array(intrinsics["K"], dtype=np.float64)
    K[0, :] *= sx
    K[1, :] *= sy
    return {**intrinsics, "K": K.tolist(), "image_size": [width, height]}


def fit_homography(src, dst, ransac=True):
    """Fit dst ~ H @ src. Returns (H normalised so H[2,2] == 1, inlier mask)."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if len(src) < 4:
        raise ValueError(f"need at least 4 point pairs, got {len(src)}")
    method = cv2.RANSAC if ransac and len(src) > 4 else 0
    H, mask = cv2.findHomography(src, dst, method, 3.0)
    if H is None:
        raise ValueError("homography fit failed")
    H = H / H[2, 2]
    mask = mask.ravel().astype(bool) if mask is not None else np.ones(len(src), bool)
    return H, mask


def apply(H, points):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.asarray(H, dtype=np.float64)).reshape(-1, 2)


def reprojection_error(H, src, dst):
    err = np.linalg.norm(apply(H, src) - np.asarray(dst, dtype=np.float64).reshape(-1, 2), axis=1)
    return float(err.mean()), float(err.max())


def css_matrix3d(H):
    """CSS matrix3d() string that maps an element's local px (x, y) through H.

    The element must use transform-origin: 0 0. CSS matrix3d is column-major.
    """
    h = np.asarray(H, dtype=np.float64)
    h = h / h[2, 2]
    vals = [
        h[0, 0], h[1, 0], 0, h[2, 0],
        h[0, 1], h[1, 1], 0, h[2, 1],
        0, 0, 1, 0,
        h[0, 2], h[1, 2], 0, h[2, 2],
    ]
    return "matrix3d(" + ",".join(f"{v:.10g}" for v in vals) + ")"


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)
