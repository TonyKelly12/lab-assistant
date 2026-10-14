"""Marker detection diagnostics.

Builds the same synthetic bench image the calibration test uses, then tries
ArUco detection with several settings and reports what each one finds.
Also saves the image so it can be inspected.

  python3 tools/diagnose_markers.py
  python3 tools/diagnose_markers.py --image calibration/capture-bench.jpg   # a real photo
"""
import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from labassist import aruco  # noqa: E402


def make_params(**overrides):
    if hasattr(cv2.aruco, "DetectorParameters") and hasattr(cv2.aruco, "ArucoDetector"):
        p = cv2.aruco.DetectorParameters()
    else:
        p = cv2.aruco.DetectorParameters_create()
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


def run(gray, params):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco.DICT_NAME))
    if hasattr(cv2.aruco, "ArucoDetector"):
        corners, ids, rejected = cv2.aruco.ArucoDetector(d, params).detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, d, parameters=params)
    found = sorted(int(i) for i in ids.flatten()) if ids is not None else []
    return found, len(rejected)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="test a real photo instead of the synthetic bench image")
    a = ap.parse_args()

    print("opencv", cv2.__version__, "| numpy", np.__version__,
          "| new aruco api", hasattr(cv2.aruco, "ArucoDetector"))

    if a.image:
        img = cv2.imread(a.image)
        if img is None:
            sys.exit(f"could not read {a.image}")
    else:
        import test_calibration as t
        H_bench_to_cam, _ = t.truth()
        img = t.bench_image(H_bench_to_cam)
        cv2.imwrite("diagnose-bench.png", img)
        print("saved the synthetic image as diagnose-bench.png (expect markers 0, 1, 2, 3)")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    print("image", gray.shape[1], "x", gray.shape[0], "| min", int(gray.min()), "max", int(gray.max()),
          "mean", round(float(gray.mean()), 1))

    trials = [
        ("labassist.aruco.detect (what the code uses)", None),
        ("defaults", {}),
        ("subpix refinement", {"cornerRefinementMethod": cv2.aruco.CORNER_REFINE_SUBPIX}),
        ("bigger threshold windows", {"adaptiveThreshWinSizeMax": 53, "adaptiveThreshWinSizeStep": 4}),
        ("looser polygon fit", {"polygonalApproxAccuracyRate": 0.06}),
        ("small markers allowed", {"minMarkerPerimeterRate": 0.01}),
    ]
    for name, overrides in trials:
        if overrides is None:
            found = sorted(aruco.detect(gray))
            print(f"  {name:45s} found {found}")
            continue
        found, rejected = run(gray, make_params(**overrides))
        print(f"  {name:45s} found {found}  (rejected candidates: {rejected})")
    for scale in (0.5, 0.25):
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        found, rejected = run(small, make_params())
        print(f"  {'defaults on image scaled x' + str(scale):45s} found {found}  (rejected candidates: {rejected})")


if __name__ == "__main__":
    main()
