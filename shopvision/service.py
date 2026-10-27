#!/usr/bin/env python3
"""Bench-vision service.  NODE: shop Pi (with the overhead camera + projector).

The "board-aware projection" brain. It owns the camera + the saved projector
calibration, so it can turn a spoken reference into marks on the real board:

    MQTT  lab/board/request  {"ref": "column 12", "id": "abcd"}
      -> snapshot the bench, find the breadboard, resolve the reference to real
         holes, and project them:
    MQTT  lab/display/bench  {items: [...]}        (the glowing holes)
    MQTT  lab/router/reply   {"id","text",...}     (a short line to speak)

Also serves HTTP /health and /board (snapshot -> found board + holes JSON, with
?ref= to preview a resolve and ?draw=1 to project it) for tuning and B-roll.

Nothing here is shop-specific in code: camera/render URLs, calibration path and
the broker login all come from the config file (copy config/shopvision.example.yaml).

Run:  python3 -m shopvision.service --config config/shopvision.yaml [--demo]
"""
import argparse
import logging
import os
import sys
import threading
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labassist import board as boardlib        # noqa: E402
from labassist import breadboard as bb          # noqa: E402
from labassist import circuit as cktlib         # noqa: E402
from labassist import config as configlib       # noqa: E402
from labassist.geometry import load_json, scale_intrinsics  # noqa: E402
from labassist.mqtt import Bus                   # noqa: E402
from labassist.net import get_image, get_json, post_json    # noqa: E402

log = logging.getLogger("shopvision")

OUTLINE_WORDS = {"", "find", "find board", "find my board", "board", "outline",
                 "where is my board", "where's my board", "scan"}
CLEAR_WORDS = {"clear", "clear board", "clear the board", "wipe", "reset", "blank"}


class BenchVision:
    def __init__(self, cfg, demo=False):
        self.cfg = cfg
        self.demo = demo
        self.camera_url = cfg["camera_url"].rstrip("/")
        self.render_url = cfg["render_url"].rstrip("/")
        self.camera = cfg.get("camera", "bench-overhead")
        self.profile_name = cfg.get("board_profile")        # None => auto by size
        self.flash = bool(cfg.get("flash", True))
        self.settle = float(cfg.get("flash_settle_s", 1.2))
        calib_path = cfg.get("output", "calibration/bench.json")
        if not os.path.exists(calib_path):
            raise SystemExit(f"no calibration at {calib_path}; run projector_calibrate.py first")
        self.calib = load_json(calib_path)
        self.intrinsics_path = cfg.get("intrinsics")
        self.last_board = None        # cached good pose for --demo fallback
        self.last_boards = []         # all boards from the last good find
        self.circuit_state = None     # {layout, boards, profiles, step} for step-through
        self.lock = threading.Lock()

    # -- geometry ------------------------------------------------------
    def _cam_to_bench(self, w, h):
        H_bc = np.array(self.calib["H_bench_to_cam"], dtype=np.float64)
        cw, ch = self.calib.get("camera_image_size", [w, h])
        if (cw, ch) != (w, h):
            H_bc = np.diag([w / cw, h / ch, 1.0]) @ H_bc
        return np.linalg.inv(H_bc)

    def snapshot(self, flash=None):
        flash = self.flash if flash is None else flash
        previous = None
        if flash:
            try:
                previous = get_json(f"{self.render_url}/state")
            except Exception:
                previous = None
            post_json(f"{self.render_url}/pattern", {"kind": "white"})
            time.sleep(self.settle)
        try:
            img = get_image(f"{self.camera_url}/snapshot/{self.camera}.jpg")
        finally:
            if flash:
                scene = previous.get("scene") if previous and previous.get("mode") == "scene" else {"items": []}
                try:
                    post_json(f"{self.render_url}/scene", scene or {"items": []})
                except Exception:
                    pass
        return img

    def find_all(self, flash=None):
        """Return every detected board (bench mm), ordered left-to-right then
        front-to-back so a layout's board indices map stably. In --demo mode,
        fall back to the last good find."""
        img = self.snapshot(flash)
        h, w = img.shape[:2]
        H_cb = self._cam_to_bench(w, h)
        intr = None
        if self.intrinsics_path and os.path.exists(self.intrinsics_path):
            intr = scale_intrinsics(load_json(self.intrinsics_path), w, h)
        area = self.calib.get("area_mm") or [-60, -60, 760, 460]
        markers = [tuple(v) for v in (self.calib.get("bench_markers") or {}).values()]
        mask = boardlib.bench_mask(img.shape, H_cb, area, exclude_mm=markers)
        found, _ = boardlib.find_boards(img, H_cb, mask, intr)
        found = sorted(found, key=lambda b: (round(b["center"][0], 0), round(b["center"][1], 0)))
        if found:
            self.last_boards = found
            self.last_board = found[0]
            return found
        if self.demo and self.last_boards:
            log.warning("no board found; using cached pose(s) (--demo)")
            return self.last_boards
        return []

    def find_board(self, flash=None):
        """Return the largest board (bench mm) or the cached one in --demo mode."""
        boards = self.find_all(flash)
        if not boards:
            return None
        return max(boards, key=lambda b: max(b["size"]))

    # -- request handling ----------------------------------------------
    def handle(self, ref, request_id=None, draw=True, flash=None):
        """Resolve a reference and (optionally) project it. Returns a result dict."""
        t = (ref or "").strip().lower()
        if t in CLEAR_WORDS:
            if draw:
                self._publish_scene({"items": []})
            return {"kind": "clear", "text": "Board cleared.", "scene": {"items": []}}

        with self.lock:
            b = self.find_board(flash)
        if b is None:
            return {"kind": "none", "text": "I can't see a board on the bench."}

        if t in OUTLINE_WORDS:
            scene = boardlib.outline_scene([b])
            text = f"Found your board, {b['size'][0]:.0f} by {b['size'][1]:.0f} millimetres."
            kind = "outline"
            resolved = None
        else:
            profile = bb.pick_profile(b, self.profile_name)
            resolved = bb.resolve(ref, b, profile)
            if resolved["kind"] == "unknown" or not resolved["holes"]:
                return {"kind": "unknown", "text": f"I couldn't place '{ref}' on the board.", "board": b}
            scene = bb.holes_scene(resolved, board=b)
            text = self._say(resolved)
            kind = resolved["kind"]

        if draw:
            self._publish_scene(scene)
        out = {"kind": kind, "text": text, "scene": scene,
               "board": {"center": b["center"], "size": b["size"], "angle": b["angle"]}}
        if resolved:
            out["resolved"] = resolved
        return out

    @staticmethod
    def _say(resolved):
        n = len(resolved["holes"])
        k = resolved["kind"]
        if k == "column":
            return f"{resolved['label']}, {n} holes."
        if k == "row":
            return f"{resolved['label']}, across the board."
        if k == "rail":
            return f"Lighting the {resolved['label']}."
        if k == "place":
            return f"Place it here: {resolved['label']}."
        if k == "hole":
            return f"Hole {resolved['label']}."
        if k == "gap":
            return "The centre gap."
        return resolved["label"]

    # -- circuit layouts -----------------------------------------------
    def handle_circuit(self, spec, request_id=None, draw=True, flash=None):
        """Project a full wiring layout onto the real holes, or step through one.

        spec: {"query"|"name": str, "step": bool?, "action": "next"|"prev"|"all"?}
        """
        action = str(spec.get("action", "")).strip().lower()
        if action in ("next", "prev", "back", "all", "step"):
            st = self.circuit_state
            if not st:
                return {"kind": "none", "text": "No circuit is up yet - ask me to wire one first."}
            total = len(st["layout"].get("wires", []))
            step = st["step"]
            if action in ("next", "step"):
                step = min(step + 1, total)
            elif action in ("prev", "back"):
                step = max(step - 1, 0)
            else:  # all
                step = total
            st["step"] = step
            scene = cktlib.render_circuit(st["layout"], st["boards"], st["profiles"], steps=step)
            if draw:
                self._publish_scene(scene)
            return {"kind": "circuit", "text": self._say_circuit(st["layout"], step, total),
                    "scene": scene, "step": step, "steps": total}

        # new circuit: resolve name first, else best-effort match the request
        layout = None
        name = str(spec.get("name", "")).strip()
        query = str(spec.get("query", "")).strip()
        if name:
            try:
                layout = cktlib.load_circuit(name)
            except FileNotFoundError:
                layout = None
        if layout is None and query:
            layout = cktlib.find_circuit(query)
        if layout is None:
            return {"kind": "unknown",
                    "text": "I don't have a circuit for that one yet. Try 'wire up an ESP32 to an LED'."}

        with self.lock:
            boards = self.find_all(flash)
        need = int(layout.get("boards", 1))
        if len(boards) < need:
            noun = "breadboard" if need == 1 else "breadboards"
            return {"kind": "none",
                    "text": f"That layout needs {need} {noun}; I can see {len(boards)}."}
        boards = boards[:need]
        profiles = cktlib.profiles_for(boards, layout, name=self.profile_name)
        problems = cktlib.validate(layout, boards, profiles)
        if problems:
            log.warning("circuit %s doesn't fit: %s", layout.get("name"), problems)
            return {"kind": "invalid",
                    "text": "That layout doesn't fit the board I see - check the board size or profile."}

        total = len(layout.get("wires", []))
        stepped = bool(spec.get("step"))
        step = 1 if (stepped and total) else total
        self.circuit_state = {"layout": layout, "boards": boards, "profiles": profiles, "step": step}
        scene = cktlib.render_circuit(layout, boards, profiles, steps=step)
        if draw:
            self._publish_scene(scene)
        if stepped:
            text = self._say_circuit(layout, step, total)
        else:
            s = "s" if total != 1 else ""
            title = layout.get("title") or layout.get("name")
            text = (f"{title}: {total} connection{s}. Components in violet, wires colour-coded. "
                    f"Say 'next' to step through, or 'clear the board' when you're done.")
        return {"kind": "circuit", "text": text, "scene": scene, "step": step, "steps": total}

    @staticmethod
    def _say_circuit(layout, step, total):
        title = layout.get("title") or layout.get("name") or "Circuit"
        if step <= 0:
            return f"{title}: components placed. Say 'next' for the first wire."
        wires = layout.get("wires", [])
        w = wires[step - 1] if 0 < step <= len(wires) else {}
        label = w.get("label") or str(w.get("net", "wire"))
        return f"Step {step} of {total}: {label}."

    def _publish_scene(self, scene):
        if self.bus:
            self.bus.publish(f"lab/display/{self.cfg.get('surface', 'bench')}", scene)
        else:
            post_json(f"{self.render_url}/scene", scene)

    # wired in main()
    bus = None


def create_app(bv):
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify(ok=True, camera=bv.camera, demo=bv.demo,
                       has_cached_board=bv.last_board is not None)

    @app.get("/board")
    def board_debug():
        ref = request.args.get("ref", "")
        draw = request.args.get("draw", "0") in ("1", "true", "yes")
        flash = request.args.get("flash")
        flash = None if flash is None else flash in ("1", "true", "yes")
        try:
            res = bv.handle(ref or "find", draw=draw, flash=flash)
        except Exception as exc:
            return jsonify(ok=False, error=str(exc)), 503
        return jsonify(ok=True, **res)

    @app.get("/circuit")
    def circuit_debug():
        spec = {k: request.args.get(k) for k in ("name", "query", "action") if request.args.get(k)}
        if request.args.get("step") in ("1", "true", "yes"):
            spec["step"] = True
        draw = request.args.get("draw", "0") in ("1", "true", "yes")
        try:
            res = bv.handle_circuit(spec, draw=draw)
        except Exception as exc:
            return jsonify(ok=False, error=str(exc)), 503
        return jsonify(ok=True, **res)

    @app.get("/circuits")
    def circuits_list():
        return jsonify(ok=True, circuits=cktlib.list_circuits())

    return app


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config/shopvision.yaml")
    p.add_argument("--demo", action="store_true", help="reuse the last good board pose if a live find fails")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = configlib.load(a.config)
    if os.environ.get("MQTT_PASS") and cfg.get("mqtt"):
        cfg["mqtt"]["password"] = os.environ["MQTT_PASS"]
    bv = BenchVision(cfg, demo=a.demo)

    bus = Bus(cfg.get("mqtt"), client_id=cfg.get("node_name", "shopvision"))
    bv.bus = bus if bus.client is not None else None

    def on_request(topic, payload):
        if isinstance(payload, dict):
            ref, rid = str(payload.get("ref", "")), payload.get("id")
            flash = payload.get("flash")
        else:
            ref, rid, flash = str(payload or ""), None, None
        try:
            res = bv.handle(ref, request_id=rid, draw=True, flash=flash)
        except Exception:
            log.exception("board request failed")
            res = {"kind": "error", "text": "Something went wrong looking at the bench."}
        if bv.bus and res.get("text"):
            bv.bus.publish("lab/router/reply", {"id": rid, "text": res["text"], "intent": "highlight_board"})
        log.info("board request %r -> %s", ref, res.get("kind"))

    def on_circuit(topic, payload):
        spec = payload if isinstance(payload, dict) else {"query": str(payload or "")}
        rid = spec.get("id")
        try:
            res = bv.handle_circuit(spec, request_id=rid, draw=True, flash=spec.get("flash"))
        except Exception:
            log.exception("circuit request failed")
            res = {"kind": "error", "text": "Something went wrong drawing that circuit."}
        if bv.bus and res.get("text"):
            bv.bus.publish("lab/router/reply", {"id": rid, "text": res["text"], "intent": "show_circuit"})
        log.info("circuit request %r -> %s",
                 spec.get("name") or spec.get("query") or spec.get("action"), res.get("kind"))

    if bv.bus:
        bus.subscribe("lab/board/request", on_request)
        bus.subscribe("lab/board/circuit", on_circuit)
        log.info("listening on lab/board/request and lab/board/circuit")
    else:
        log.warning("MQTT disabled; HTTP /board and /circuit only")

    http = cfg.get("http", {})
    create_app(bv).run(host=http.get("host", "0.0.0.0"), port=int(http.get("port", 8070)), threaded=True)


if __name__ == "__main__":
    main()
