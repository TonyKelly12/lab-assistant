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


def detect(gray, scales=None):
    """Detect markers in a grayscale (or BGR) image.

    Returns {id: corners} where corners is a (4, 2) float32 array in
    full-resolution pixels, marker order: top-left, top-right, bottom-right,
    bottom-left.

    Big images are also searched at reduced sizes: OpenCV 4.6 (Raspberry Pi
    OS Bookworm) misses markers in full-resolution camera frames that it finds
    at half size. Markers found only at a reduced size get their corners
    refined on the full-resolution image, so accuracy is kept.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if scales is None:
        longest = max(gray.shape)
        scales = [1.0] + [s for s in (0.5, 0.25) if longest * s >= 600]
    found = {}
    # Second pass on a locally contrast-boosted copy: helps low-contrast
    # markers (e.g. laser-engraved wood, where the burn is mid-brown with
    # grain showing through). Corners are always refined on the original.
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    for source in (gray, enhanced):
        for scale in scales:
            if scale == 1.0:
                img = source
            else:
                img = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            for marker_id, corners in _detect_once(img).items():
                if marker_id in found:
                    continue
                if scale != 1.0:
                    # Pixel centres: small (u) <-> full ((u + 0.5) / scale - 0.5).
                    corners = _refine(gray, (corners + 0.5) / scale - 0.5, scale)
                found[marker_id] = corners
    return found


def _refine(gray, corners, scale):
    """Sharpen corners found on a downscaled image against the full-resolution one."""
    win = max(2, int(round(2 / scale)))
    pts = corners.reshape(-1, 1, 2).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    cv2.cornerSubPix(gray, pts, (win, win), (-1, -1), criteria)
    return pts.reshape(4, 2)


def _detect_once(gray):
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
