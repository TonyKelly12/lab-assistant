"""Projector calibration: work out how bench millimetres map to projector pixels.

Before running:
  - camera node and render node services are running
  - the four printed bench markers are taped down and their centres measured
    into config/bench.yaml
  - the room is fairly dark and the bench is mostly clear

Full calibration (first time, or after the camera moves relative to the projector):
  python3 calibration/projector_calibrate.py --config config/bench.yaml

Bench-only (the tripod got bumped but camera and projector moved together):
  python3 calibration/projector_calibrate.py --config config/bench.yaml --bench-only

Steps: projector goes white -> camera finds the printed bench markers ->
projector shows a marker grid -> camera finds it -> both mappings combined and
sent to the render node, which then shows a check pattern (crosses should land
on the printed markers).
"""
import argparse
import datetime
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labassist import calib  # noqa: E402
from labassist import config as configlib  # noqa: E402
from labassist.geometry import load_json, save_json, scale_intrinsics  # noqa: E402
from labassist.net import get_image, get_json, post_json, post_png  # noqa: E402


def require_service(name, url, start_cmd, check=None):
    """Exit with a plain message if a service isn't answering."""
    try:
        info = get_json(f"{url}/health", timeout=5)
    except Exception as exc:
        sys.exit(f"The {name} isn't answering at {url} ({exc}).\n"
                 f"Start it in another terminal and leave it running:\n  {start_cmd}")
    if check:
        check(info)
    return info


def check_scene(bench_markers, area):
    x0, y0, x1, y1 = area
    items = [{"type": "rect", "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0, "stroke": "#00e5ff", "width": 2}]
    step = 50
    for x in np.arange(np.ceil(x0 / step) * step, x1, step):
        items.append({"type": "line", "x1": float(x), "y1": y0, "x2": float(x), "y2": y1, "stroke": "#235", "width": 0.6})
    for y in np.arange(np.ceil(y0 / step) * step, y1, step):
        items.append({"type": "line", "x1": x0, "y1": float(y), "x2": x1, "y2": float(y), "stroke": "#235", "width": 0.6})
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    for marker_id, (x, y) in bench_markers.items():
        items.append({"type": "cross", "x": x, "y": y, "size": 40, "stroke": "#ffeb3b", "width": 1.5})
        # Label on the inside of the marker so it stays on the bench.
        lx = x + (45 if x < cx else -45)
        ly = y + (45 if y < cy else -38)
        items.append({"type": "text", "x": lx, "y": ly, "text": f"ID {marker_id}", "size": 10,
                      "color": "#ffeb3b", "anchor": "start" if x < cx else "end"})
    items.append({"type": "text", "x": cx, "y": cy, "text": "Crosses should sit on the printed markers",
                  "size": 14, "anchor": "middle", "color": "#ffffff"})
    return {"items": items}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config/bench.yaml")
    p.add_argument("--bench-only", action="store_true", help="reuse the saved camera->projector mapping")
    p.add_argument("--settle", type=float, default=2.0, help="seconds to wait after changing the projection")
    p.add_argument("--save-images", action="store_true", help="keep the captured photos for debugging")
    a = p.parse_args()

    cfg = configlib.load(a.config)
    cam_url = cfg["camera_url"].rstrip("/")
    render_url = cfg["render_url"].rstrip("/")
    camera = cfg.get("camera", "bench-overhead")
    proj_w, proj_h = cfg.get("projector_size", [1920, 1080])
    bench_markers = {int(k): tuple(map(float, v)) for k, v in cfg["bench_markers"].items()}
    area = calib.bench_area(bench_markers, cfg.get("margin_mm", 60))
    out_path = cfg.get("output", "calibration/bench.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    snapshot_url = f"{cam_url}/snapshot/{camera}.jpg"

    def camera_listed(info):
        if camera not in info.get("cameras", []):
            sys.exit(f"The camera service is running but has no camera named {camera!r} "
                     f"(it has {info.get('cameras')}). Fix 'camera' in {a.config}.")

    require_service("camera service", cam_url,
                    "python3 -m camera_node.service --config config/camera-node.yaml", camera_listed)
    render_info = require_service("render server", render_url,
                                  "python3 -m render_node.server --config config/render-node.yaml")
    if not render_info.get("viewers"):
        sys.exit("The render server is running, but no projector page is connected to it.\n"
                 "Open the page on the projector first:\n  DISPLAY=:0 ./render_node/kiosk.sh")

    def capture(label):
        time.sleep(a.settle)
        img = get_image(snapshot_url)
        if a.save_images:
            cv2.imwrite(f"calibration/capture-{label}.jpg", img)
        return img

    intrinsics = None
    if cfg.get("intrinsics") and os.path.exists(cfg["intrinsics"]):
        intrinsics = load_json(cfg["intrinsics"])
        print("using lens calibration", cfg["intrinsics"])
    else:
        print("no lens calibration found; continuing without it (run lens_calibrate.py for better accuracy)")

    # 1. Printed bench markers, lit by a plain white projection so they show up
    #    even with the room lights off.
    print("1/3 projector white, looking for the printed bench markers...")
    post_json(f"{render_url}/pattern", {"kind": "white"})
    img = capture("bench")
    intr = scale_intrinsics(intrinsics, img.shape[1], img.shape[0])
    try:
        H_bench_to_cam, bench_report = calib.solve_bench_to_cam(img, bench_markers, intr)
    except ValueError as exc:
        sys.exit(f"   failed: {exc}\n   Re-run with --save-images and look at calibration/capture-bench.jpg")
    print("   found markers", bench_report["markers_seen"])

    # 2. Camera -> projector, from a projected marker grid (or the saved one).
    if a.bench_only:
        if not os.path.exists(out_path):
            sys.exit(f"--bench-only needs an earlier full calibration in {out_path}")
        H_cam_to_proj = np.array(load_json(out_path)["H_cam_to_proj"])
        proj_report = {"reused": True}
        print("2/3 reusing saved camera->projector mapping")
    else:
        print("2/3 projecting marker grid...")
        pattern, layout = calib.projector_pattern(proj_w, proj_h)
        post_png(f"{render_url}/pattern", pattern)
        img = capture("pattern")
        try:
            H_cam_to_proj, proj_report = calib.solve_cam_to_proj(img, layout, intr)
        except ValueError as exc:
            post_json(f"{render_url}/pattern", {"kind": "black"})
            sys.exit(f"   failed: {exc}\n   Re-run with --save-images and look at calibration/capture-pattern.jpg")
        print(f"   saw {proj_report['markers_seen']}/{proj_report['markers_total']} markers, "
              f"mean error {proj_report['mean_error_px']} px")

    # 3. Combine and send to the render node.
    H_bench_to_proj = H_cam_to_proj @ H_bench_to_cam
    H_bench_to_proj /= H_bench_to_proj[2, 2]
    result = {
        "surface": cfg.get("surface", "bench"),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "area_mm": area,
        "bench_markers": {str(k): v for k, v in bench_markers.items()},
        "projector_size": [proj_w, proj_h],
        "camera_image_size": [img.shape[1], img.shape[0]],
        "H_bench_to_cam": H_bench_to_cam.tolist(),
        "H_cam_to_proj": H_cam_to_proj.tolist(),
        "H_bench_to_proj": H_bench_to_proj.tolist(),
        "report": {"bench": bench_report, "projector": proj_report},
    }
    save_json(out_path, result)
    post_json(f"{render_url}/calibration", result)
    post_json(f"{render_url}/scene", check_scene(bench_markers, area))
    print(f"3/3 saved {out_path} and sent it to the render node.")
    print("Check the bench: the yellow crosses should sit on the printed markers' centres.")


if __name__ == "__main__":
    main()
