"""Find the breadboard on the bench from a camera snapshot - no marker needed.

Uses the saved projector calibration (calibration/bench.json) to turn camera
pixels into bench millimetres, finds bright rectangles (white breadboards),
prints their size, position and angle, saves a debug picture, and projects
the outline back onto the bench so you can see if it lines up.

  python3 tools/find_board.py --config config/bench.yaml            # flash white, find, show outline
  python3 tools/find_board.py --config config/bench.yaml --no-flash # use the room light as it is
  python3 tools/find_board.py --config config/bench.yaml --image snap.jpg --no-show   # offline

Positions are bench mm from marker 0's centre (x right, y toward you).
A half-size breadboard is about 55 x 83 mm.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labassist import board  # noqa: E402
from labassist import config as configlib  # noqa: E402
from labassist.geometry import load_json, scale_intrinsics  # noqa: E402
from labassist.net import get_image, get_json, post_json  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config/bench.yaml")
    p.add_argument("--image", help="use this photo instead of taking a snapshot")
    p.add_argument("--no-flash", action="store_true", help="don't light the bench white for the photo")
    p.add_argument("--no-show", action="store_true", help="don't project the outline")
    p.add_argument("--settle", type=float, default=1.5, help="seconds to wait after the white flash")
    p.add_argument("--debug-out", default="board-find.jpg", help="where to save the annotated photo")
    a = p.parse_args()

    cfg = configlib.load(a.config)
    cam_url = cfg["camera_url"].rstrip("/")
    render_url = cfg["render_url"].rstrip("/")
    camera = cfg.get("camera", "bench-overhead")
    calib_path = cfg.get("output", "calibration/bench.json")
    if not os.path.exists(calib_path):
        sys.exit(f"No calibration at {calib_path}; run calibration/projector_calibrate.py first.")
    calib = load_json(calib_path)
    H_bench_to_cam = np.array(calib["H_bench_to_cam"], dtype=np.float64)

    # 1. Photo (optionally with the projector flashing white so the bench is evenly lit).
    if a.image:
        img = cv2.imread(a.image)
        if img is None:
            sys.exit(f"could not read {a.image}")
    else:
        previous = None
        if not a.no_flash:
            try:
                previous = get_json(f"{render_url}/state")
            except Exception:
                previous = None
            post_json(f"{render_url}/pattern", {"kind": "white"})
            time.sleep(a.settle)
        try:
            img = get_image(f"{cam_url}/snapshot/{camera}.jpg")
        finally:
            if not a.no_flash:
                if previous and previous.get("mode") == "scene":
                    post_json(f"{render_url}/scene", previous.get("scene") or {"items": []})
                else:
                    post_json(f"{render_url}/scene", {"items": []})

    # 2. Calibration was made at one image size; rescale if this photo differs.
    h, w = img.shape[:2]
    cw, ch = calib.get("camera_image_size", [w, h])
    if (cw, ch) != (w, h):
        S = np.diag([w / cw, h / ch, 1.0])
        H_bench_to_cam = S @ H_bench_to_cam
    H_cam_to_bench = np.linalg.inv(H_bench_to_cam)
    intrinsics = None
    if cfg.get("intrinsics") and os.path.exists(cfg["intrinsics"]):
        intrinsics = scale_intrinsics(load_json(cfg["intrinsics"]), w, h)

    # 3. Find bright rectangles on the bench, ignoring the marker plates.
    area = calib.get("area_mm") or [-60, -60, 760, 460]
    markers = [tuple(v) for v in (calib.get("bench_markers") or {}).values()]
    mask = board.bench_mask(img.shape, H_cam_to_bench, area, exclude_mm=markers)
    found, thresh = board.find_boards(img, H_cam_to_bench, mask, intrinsics)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"photo {w}x{h}, mean brightness {gray.mean():.0f}/255, threshold {thresh:.0f}")
    if not found:
        print("No board-sized bright rectangle found. Try again with the flash (default), "
              "or check the board is inside the marked area and not touching other bright things.")
    for n, b in enumerate(found, 1):
        hb = board.HALF_BREADBOARD_MM
        note = ""
        if abs(b["size"][0] - hb[0]) < 6 and abs(b["size"][1] - hb[1]) < 8:
            note = f"   <- looks like a half-size breadboard ({hb[0]:.0f} x {hb[1]:.0f})"
        print(f"#{n}: centre ({b['center'][0]:.1f}, {b['center'][1]:.1f}) mm, "
              f"size {b['size'][0]:.1f} x {b['size'][1]:.1f} mm, long side at {b['angle']:.1f} deg, "
              f"solidity {b['fill']:.2f}{note}")
        print("    corners mm: " + "  ".join(f"({x:.0f}, {y:.0f})" for x, y in b["corners"]))

    # 4. Debug picture and projected outline.
    vis = img.copy()
    for n, b in enumerate(found, 1):
        pts = np.array(b["px"], np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 3)
        cv2.putText(vis, f"#{n}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    if gray.mean() < 60:
        vis = cv2.convertScaleAbs(vis, alpha=min(4.0, 90.0 / max(gray.mean(), 1)))
    cv2.imwrite(a.debug_out, vis)
    print(f"saved {a.debug_out}")
    if found and not a.no_show and not a.image:
        post_json(f"{render_url}/scene", board.outline_scene(found))
        print("Outline projected: the cyan box should sit on the breadboard's edges.")


if __name__ == "__main__":
    main()
