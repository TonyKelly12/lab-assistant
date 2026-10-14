"""Camera node service.

Runs on the camera Pi. For every configured camera it:
  - serves full-resolution JPEG snapshots over HTTP (for the AI and calibration)
  - optionally tracks ArUco markers and motion on a low-res stream, and
    publishes changes over MQTT (when a broker is configured)

Run:  python3 -m camera_node.service --config config/camera-node.yaml
"""
import argparse
import logging
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, request

from labassist import aruco
from labassist import config as configlib
from labassist.mqtt import Bus

from .sources import open_source

log = logging.getLogger("camera-node")


class Camera:
    def __init__(self, name, cfg, bus):
        self.name = name
        self.cfg = cfg
        self.bus = bus
        self.source = open_source(cfg)
        self.track = bool(cfg.get("track", False))
        self.markers = {}          # id -> corners in full-resolution pixels
        self.markers_time = 0.0
        self.motion_active = False
        self.last_motion = 0.0
        self.full_size = getattr(self.source, "main_size", None)
        if self.track:
            threading.Thread(target=self._track_loop, daemon=True).start()

    def snapshot_jpeg(self, max_width=None, quality=90):
        frame = self.source.snapshot()
        if max_width and frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            frame = cv2.resize(frame, (max_width, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return buf.tobytes()

    def _scale(self, small):
        if self.full_size is None:
            h, w = self.source.snapshot().shape[:2]
            self.full_size = (w, h)
        return self.full_size[0] / small.shape[1], self.full_size[1] / small.shape[0]

    def _track_loop(self):
        interval = 1.0 / float(self.cfg.get("track_hz", 5))
        motion_threshold = float(self.cfg.get("motion_fraction", 0.005))
        idle_after = float(self.cfg.get("idle_after_s", 10))
        prev = None
        last_marker_ids = None
        while True:
            start = time.time()
            try:
                small = self.source.frame_small()
            except Exception as exc:
                log.warning("%s: frame failed: %s", self.name, exc)
                time.sleep(2)
                continue
            sx, sy = self._scale(small)

            # Markers, reported in full-resolution pixel coordinates.
            found = aruco.detect(small)
            self.markers = {mid: (c * [sx, sy]).tolist() for mid, c in found.items()}
            self.markers_time = start
            ids = sorted(found)
            if ids != last_marker_ids:
                self.bus.publish(f"lab/camera/{self.name}/markers", self.marker_payload(), retain=True)
                last_marker_ids = ids

            # Motion: share of pixels that changed since the last frame.
            blurred = cv2.GaussianBlur(small, (9, 9), 0)
            if prev is not None:
                changed = np.count_nonzero(cv2.absdiff(blurred, prev) > 25) / blurred.size
                if changed > motion_threshold:
                    self.last_motion = start
                    if not self.motion_active:
                        self.motion_active = True
                        self.bus.publish(f"lab/camera/{self.name}/motion", {"active": True, "t": start}, retain=True)
                elif self.motion_active and start - self.last_motion > idle_after:
                    self.motion_active = False
                    self.bus.publish(f"lab/camera/{self.name}/motion", {"active": False, "t": start}, retain=True)
            prev = blurred
            time.sleep(max(0.0, interval - (time.time() - start)))

    def marker_payload(self):
        return {
            "camera": self.name,
            "t": self.markers_time,
            "image_size": list(self.full_size) if self.full_size else None,
            "markers": {str(k): v for k, v in self.markers.items()},
        }


def create_app(cameras):
    app = Flask(__name__)

    def get(name):
        cam = cameras.get(name)
        if cam is None:
            abort(404, f"no camera named {name}")
        return cam

    @app.get("/health")
    def health():
        return jsonify(ok=True, cameras=list(cameras))

    @app.get("/cameras")
    def list_cameras():
        return jsonify({
            n: {"type": c.cfg.get("type"), "track": c.track, "full_size": c.full_size}
            for n, c in cameras.items()
        })

    @app.get("/snapshot/<name>.jpg")
    def snapshot(name):
        cam = get(name)
        try:
            data = cam.snapshot_jpeg(
                max_width=request.args.get("max_width", type=int),
                quality=request.args.get("q", 90, type=int),
            )
        except Exception as exc:
            abort(503, str(exc))
        return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/markers/<name>")
    def markers(name):
        cam = get(name)
        if not cam.track:
            # Not tracking continuously: detect once on a fresh snapshot.
            frame = cam.source.snapshot()
            found = aruco.detect(frame)
            h, w = frame.shape[:2]
            return jsonify(camera=name, t=time.time(), image_size=[w, h],
                           markers={str(k): v.tolist() for k, v in found.items()})
        return jsonify(cam.marker_payload())

    @app.get("/motion/<name>")
    def motion(name):
        cam = get(name)
        return jsonify(active=cam.motion_active, last_motion=cam.last_motion)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/camera-node.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = configlib.load(args.config)
    bus = Bus(cfg.get("mqtt"), client_id=cfg.get("node_name", "camera-node"))
    cameras = {}
    for name, cam_cfg in (cfg.get("cameras") or {}).items():
        if cam_cfg.get("enabled", True) is False:
            log.info("camera %s disabled in config; skipping", name)
            continue
        try:
            cameras[name] = Camera(name, cam_cfg, bus)
        except Exception:
            log.exception("camera %s failed to start; skipping it", name)
    if not cameras:
        raise SystemExit("no cameras started; check the config and `rpicam-hello --list-cameras` / `v4l2-ctl --list-devices`")

    http = cfg.get("http", {})
    create_app(cameras).run(host=http.get("host", "0.0.0.0"), port=int(http.get("port", 8080)), threaded=True)


if __name__ == "__main__":
    main()
