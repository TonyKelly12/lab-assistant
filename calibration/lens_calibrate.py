"""Lens calibration for the overhead camera (removes its fisheye-ish bend).

Hold the printed checkerboard flat in view, press Enter to capture, move it to
a new spot/angle, repeat. Cover the centre, all four corners and some tilted
views. 15 good captures is plenty.

Run (from any machine on the network):
  python3 calibration/lens_calibrate.py --camera-url http://bench-cam.local:8080 --camera bench-overhead
Or from saved photos:
  python3 calibration/lens_calibrate.py --images photos/*.jpg
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labassist.geometry import save_json  # noqa: E402
from labassist.net import get_image  # noqa: E402


def find_corners(img, pattern):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not ok:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera-url", help="camera node, e.g. http://bench-cam.local:8080")
    p.add_argument("--camera", default="bench-overhead")
    p.add_argument("--images", nargs="*", help="use saved photos instead of the camera")
    p.add_argument("--cols", type=int, default=9, help="inner corners across")
    p.add_argument("--rows", type=int, default=6, help="inner corners down")
    p.add_argument("--square-mm", type=float, default=25)
    p.add_argument("--count", type=int, default=15)
    p.add_argument("--out", default="calibration/intrinsics-bench-overhead.json")
    a = p.parse_args()

    pattern = (a.cols, a.rows)
    obj = np.zeros((a.cols * a.rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square_mm
    obj_points, img_points, size = [], [], None

    def add(img, label):
        nonlocal size
        corners = find_corners(img, pattern)
        if corners is None:
            print(f"  {label}: checkerboard NOT found, skipped")
            return
        size = (img.shape[1], img.shape[0])
        obj_points.append(obj)
        img_points.append(corners)
        print(f"  {label}: ok ({len(img_points)} good)")

    if a.images:
        paths = [f for pat in a.images for f in glob.glob(pat)]
        for path in paths:
            img = cv2.imread(path)
            if img is not None:
                add(img, path)
    elif a.camera_url:
        url = f"{a.camera_url.rstrip('/')}/snapshot/{a.camera}.jpg"
        os.makedirs("calibration/lens-photos", exist_ok=True)
        while len(img_points) < a.count:
            cmd = input(f"[{len(img_points)}/{a.count}] Position the board, Enter to capture (q to finish): ")
            if cmd.strip().lower() == "q":
                break
            img = get_image(url)
            cv2.imwrite(f"calibration/lens-photos/{len(img_points):02d}.jpg", img)
            add(img, "capture")
    else:
        p.error("give --camera-url or --images")

    if len(img_points) < 6:
        sys.exit(f"only {len(img_points)} good views; need at least 6 (15 is better)")

    rms, K, D, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    save_json(a.out, {"image_size": list(size), "K": K.tolist(), "D": D.ravel().tolist(),
                      "rms_px": rms, "views": len(img_points)})
    print(f"\nRMS reprojection error: {rms:.3f} px  (under 1.0 is good, under 0.5 is great)")
    print("saved", a.out)


if __name__ == "__main__":
    main()
