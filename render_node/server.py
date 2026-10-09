"""Render node server.

Runs on the Pi that drives a projector. Chromium (kiosk mode) shows
http://localhost:8090/ fullscreen on the projector; this server pushes
what to draw to that page.

Two modes:
  pattern - raw full-screen image in projector pixels (used by calibration)
  scene   - shapes in bench millimetres, warped onto the bench by the
            calibration homography

Inputs: HTTP (POST /scene, /pattern, /calibration) and, when a broker is
configured, MQTT topics lab/display/<surface> and lab/display/<surface>/pattern.

Run:  python3 -m render_node.server --config config/render-node.yaml
"""
import argparse
import json
import logging
import os
import queue
import threading
import time

from flask import Flask, Response, abort, jsonify, request, send_from_directory

from labassist import config as configlib
from labassist.geometry import load_json, save_json
from labassist.mqtt import Bus

log = logging.getLogger("render-node")
STATIC = os.path.join(os.path.dirname(__file__), "static")


class State:
    def __init__(self, surface, projector_size, calibration_path):
        self.lock = threading.Lock()
        self.surface = surface
        self.projector_size = projector_size
        self.calibration_path = calibration_path
        self.calibration = None
        if calibration_path and os.path.exists(calibration_path):
            self.calibration = load_json(calibration_path)
            log.info("loaded calibration from %s", calibration_path)
        self.mode = "scene"
        self.scene = {"items": []}
        self.pattern = {"kind": "black"}
        self.pattern_png = None
        self.version = 0
        self.listeners = []

    def snapshot(self):
        return {
            "version": self.version,
            "surface": self.surface,
            "projector_size": self.projector_size,
            "mode": self.mode,
            "scene": self.scene,
            "pattern": self.pattern,
            "calibration": self.calibration,
        }

    def _changed(self):
        self.version += 1
        data = json.dumps(self.snapshot())
        for q in list(self.listeners):
            try:
                q.put_nowait(data)
            except queue.Full:
                pass

    def set_scene(self, scene):
        with self.lock:
            if scene.get("append"):
                self.scene = {"items": self.scene.get("items", []) + scene.get("items", [])}
            else:
                self.scene = {"items": scene.get("items", [])}
            self.mode = "scene"
            self._changed()

    def set_pattern(self, pattern, png=None):
        with self.lock:
            if png is not None:
                self.pattern_png = png
                pattern = {"kind": "image", "url": f"/pattern.png?v={self.version + 1}"}
            self.pattern = pattern
            self.mode = "pattern"
            self._changed()

    def set_calibration(self, calibration):
        with self.lock:
            self.calibration = calibration
            if self.calibration_path:
                save_json(self.calibration_path, calibration)
            self._changed()


def create_app(state):
    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/static/<path:name>")
    def static_file(name):
        return send_from_directory(STATIC, name)

    @app.get("/favicon.ico")
    def favicon():
        return Response(status=204)

    @app.get("/health")
    def health():
        return jsonify(ok=True, surface=state.surface, calibrated=state.calibration is not None,
                       mode=state.mode, viewers=len(state.listeners))

    @app.get("/state")
    def get_state():
        with state.lock:
            return jsonify(state.snapshot())

    @app.get("/events")
    def events():
        q = queue.Queue(maxsize=20)
        with state.lock:
            state.listeners.append(q)
            first = json.dumps(state.snapshot())

        def stream():
            try:
                yield f"data: {first}\n\n"
                while True:
                    try:
                        yield f"data: {q.get(timeout=15)}\n\n"
                    except queue.Empty:
                        yield ": keepalive\n\n"
            finally:
                with state.lock:
                    if q in state.listeners:
                        state.listeners.remove(q)

        return Response(stream(), mimetype="text/event-stream", headers={"Cache-Control": "no-store"})

    @app.post("/scene")
    def post_scene():
        scene = request.get_json(force=True, silent=True)
        if not isinstance(scene, dict):
            abort(400, "expected a JSON object with an 'items' list")
        state.set_scene(scene)
        return jsonify(ok=True, version=state.version)

    @app.post("/pattern")
    def post_pattern():
        if request.mimetype == "image/png":
            state.set_pattern(None, png=request.get_data())
        else:
            pattern = request.get_json(force=True, silent=True) or {}
            if pattern.get("kind") not in ("black", "white"):
                abort(400, "kind must be black or white, or POST an image/png body")
            state.set_pattern(pattern)
        return jsonify(ok=True, version=state.version)

    @app.get("/pattern.png")
    def pattern_png():
        if state.pattern_png is None:
            abort(404)
        return Response(state.pattern_png, mimetype="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/calibration")
    def get_calibration():
        if state.calibration is None:
            abort(404, "not calibrated yet")
        return jsonify(state.calibration)

    @app.post("/calibration")
    def post_calibration():
        cal = request.get_json(force=True, silent=True)
        if not isinstance(cal, dict) or "H_bench_to_proj" not in cal or "area_mm" not in cal:
            abort(400, "expected H_bench_to_proj and area_mm")
        state.set_calibration(cal)
        return jsonify(ok=True)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/render-node.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = configlib.load(args.config)
    surface = cfg.get("surface", "bench")
    state = State(surface, cfg.get("projector_size", [1920, 1080]), cfg.get("calibration_file", "calibration/bench.json"))

    bus = Bus(cfg.get("mqtt"), client_id=cfg.get("node_name", f"render-{surface}"))
    bus.subscribe(f"lab/display/{surface}", lambda topic, msg: state.set_scene(msg or {}))
    bus.subscribe(f"lab/display/{surface}/pattern", lambda topic, msg: state.set_pattern(msg or {"kind": "black"}))

    http = cfg.get("http", {})
    log.info("open http://localhost:%s/ fullscreen on the projector", http.get("port", 8090))
    create_app(state).run(host=http.get("host", "0.0.0.0"), port=int(http.get("port", 8090)), threaded=True)


if __name__ == "__main__":
    main()
