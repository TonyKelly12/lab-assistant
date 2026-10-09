"""ArUco helpers that work across OpenCV versions.

Raspberry Pi OS ships different OpenCV versions depending on release; the
ArUco API changed in 4.7 (ArucoDetector class). These wrappers hide that.
"""
import cv2
import numpy as np

DICT_NAME = "DICT_4X4_100"

# ID ranges keep printed and projected markers from ever being confused.
BENCH_MARKER_IDS = range(0, 10)      # printed, taped to the bench
BOARD_MARKER_IDS = range(10, 20)     # printed, stuck on breadboards (later)
PROJECTED_MARKER_IDS = range(20, 100)  # drawn by the projector during calibration


def _dictionary():
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))


def detect(gray):
    """Detect markers in a grayscale (or BGR) image.

    Returns {id: corners} where corners is a (4, 2) float32 array in
    marker order: top-left, top-right, bottom-right, bottom-left.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    dictionary = _dictionary()
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:  # OpenCV < 4.7
        params = cv2.aruco.DetectorParameters_create()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    found = {}
    if ids is None:
        return found
    for marker_id, c in zip(ids.flatten(), corners):
        found[int(marker_id)] = c.reshape(4, 2).astype(np.float32)
    return found


def marker_image(marker_id, side_px):
    """Return a black-on-white marker image (uint8), side_px square, no border."""
    dictionary = _dictionary()
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)
    return cv2.aruco.drawMarker(dictionary, marker_id, side_px)  # OpenCV < 4.7


def center(corners):
    return corners.mean(axis=0)
