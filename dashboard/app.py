#!/usr/bin/env python3
"""Lab console: a tiny status + launch UI for the lab-assistant services.

Run one instance per node. Each instance controls its own local systemd units
and health-pings whatever remotes you list. One instance acts as the hub: give
it `peers` and it aggregates their status and proxies control to them, so a
single page (phone or a wall tablet) shows and drives every node.

  GET  /                      the console page
  GET  /api/status[?local=1]  this node (+peers unless local=1): services + remotes
  POST /api/control           {node, unit, action in start|stop|restart}  [needs key]
  GET  /api/demo              the storyboard beats (from the demo script)
  POST /api/demo/fire         {cmd}  publish to the router as if spoken        [needs key]
  GET  /health

Control actions need the shared key (header X-Dash-Key), set via DASHBOARD_KEY
or the config. Status/health are read-only and open on the LAN. Bind this to
your LAN only; it can start and stop services.

Run:  python3 -m dashboard.app --config config/dashboard.yaml
"""
import argparse
import hmac
import json
import logging
import os
import subprocess
import sys
import time
import uuid

import requests
from flask import Flask, Response, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labassist import config as configlib  # noqa: E402
from labassist.mqtt import Bus  # noqa: E402

log = logging.getLogger("lab-console")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
ACTIONS = {"start", "stop", "restart"}


# ---------------------------------------------------------------- systemd
def parse_show(text):
    d = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            d[k] = v
    return d


def service_state(unit):
    """One systemd unit's state as a small dict (no exception leaks)."""
    try:
        out = subprocess.run(
            ["systemctl", "show", unit, "-p", "ActiveState", "-p", "SubState",
             "-p", "UnitFileState", "-p", "Result", "-p", "LoadState"],
            capture_output=True, text=True, timeout=5,
        )
        d = parse_show(out.stdout)
    except Exception as e:  # noqa: BLE001
        return {"unit": unit, "state": "unknown", "detail": str(e)}
    load = d.get("LoadState", "")
    active = d.get("ActiveState", "")
    if load not in ("loaded",):
        state = "missing"
    elif active == "active":
        state = "active"
    elif active == "failed" or d.get("Result") not in ("success", "", None):
        state = "failed"
    else:
        state = "inactive"
    return {
        "unit": unit, "state": state,
        "sub": d.get("SubState", ""),
        "enabled": d.get("UnitFileState", "") in ("enabled", "enabled-runtime"),
    }


def run_systemctl(action, unit):
    try:
        r = subprocess.run(["sudo", "-n", "systemctl", action, unit],
                           capture_output=True, text=True, timeout=30)
    except Exception as e:  # noqa: BLE001
        return False, str(e)
    if r.returncode != 0:
        return False, (r.stderr or r.stdout or f"exit {r.returncode}").strip()
    return True, "ok"


# ---------------------------------------------------------------- remotes
def ping(url, timeout=3):
    try:
        r = requests.get(url, timeout=timeout)
        return {"ok": r.status_code < 400, "code": r.status_code}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "code": None, "detail": str(e)[:120]}


# ---------------------------------------------------------------- app
def create_app(cfg):
    app = Flask(__name__)
    node = cfg.get("node_name", "node")
    services = list(cfg.get("services") or [])
    remotes = list(cfg.get("remotes") or [])
    peers = list(cfg.get("peers") or [])
    require_key = cfg.get("require_key", True)
    key = os.environ.get("DASHBOARD_KEY") or cfg.get("key") or ""
    demo_path = cfg.get("demo_script", "config/demo_script.yaml")

    bus = Bus(cfg.get("mqtt"), client_id=f"lab-console-{node}")
    app._bus = bus if bus.client is not None else None

    def authed():
        if not require_key:
            return True
        if not key:
            return False  # fail closed: require_key but none set
        return hmac.compare_digest(request.headers.get("X-Dash-Key", ""), key)

    def local_block():
        return {
            "node": node,
            "controllable": True,
            "services": [service_state(u) for u in services],
            "remotes": [{"name": r["name"], **ping(r["url"])} for r in remotes],
        }

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/health")
    def health():
        return jsonify(ok=True, node=node)

    @app.get("/api/status")
    def status():
        blocks = [local_block()]
        if request.args.get("local") not in ("1", "true"):
            for p in peers:
                try:
                    headers = {"X-Dash-Key": p.get("key", key)} if (p.get("key") or key) else {}
                    r = requests.get(f"{p['url'].rstrip('/')}/api/status?local=1",
                                     headers=headers, timeout=4)
                    blk = r.json()
                    blk["node"] = blk.get("node") or p["name"]
                    blk["peer_url"] = p["url"]
                    blocks.append(blk)
                except Exception as e:  # noqa: BLE001
                    blocks.append({"node": p["name"], "controllable": False,
                                   "offline": True, "detail": str(e)[:120],
                                   "services": [], "remotes": []})
        return jsonify(nodes=blocks, generated_at=time.strftime("%H:%M:%S"))

    @app.post("/api/control")
    def control():
        if not authed():
            return jsonify(ok=False, error="bad or missing key"), 401
        body = request.get_json(force=True, silent=True) or {}
        target, unit, action = body.get("node", node), body.get("unit"), body.get("action")
        if action not in ACTIONS:
            return jsonify(ok=False, error="action must be start/stop/restart"), 400
        if target == node:
            if unit not in services:
                return jsonify(ok=False, error=f"{unit} not managed here"), 403
            ok, detail = run_systemctl(action, unit)
            return jsonify(ok=ok, detail=detail), (200 if ok else 500)
        # proxy to the peer that owns it
        for p in peers:
            if p["name"] == target:
                try:
                    r = requests.post(f"{p['url'].rstrip('/')}/api/control",
                                      headers={"X-Dash-Key": p.get("key", key), "Content-Type": "application/json"},
                                      data=json.dumps({"node": target, "unit": unit, "action": action}),
                                      timeout=35)
                    return Response(r.content, status=r.status_code, mimetype="application/json")
                except Exception as e:  # noqa: BLE001
                    return jsonify(ok=False, error=f"peer unreachable: {e}"), 502
        return jsonify(ok=False, error=f"unknown node {target}"), 404

    @app.get("/api/demo")
    def demo_beats():
        try:
            data = configlib.load(demo_path)
            steps = data.get("steps") if isinstance(data, dict) else data
        except Exception:
            steps = None
        if not steps:
            steps = [{"beat": "Cold open", "say": "Turn on the bench lights", "cmd": "turn on the bench lights"},
                     {"beat": "Reveal", "say": "Find my board", "cmd": "find my board"},
                     {"beat": "Hero", "say": "Light up column 12", "cmd": "light up column 12"},
                     {"beat": "Hero", "say": "Highlight the power rails", "cmd": "highlight the power rails"}]
        return jsonify(beats=steps)

    @app.post("/api/demo/fire")
    def demo_fire():
        if not authed():
            return jsonify(ok=False, error="bad or missing key"), 401
        cmd = (request.get_json(force=True, silent=True) or {}).get("cmd", "").strip()
        if not cmd:
            return jsonify(ok=False, error="empty command"), 400
        if app._bus is None:
            return jsonify(ok=False, error="MQTT not available on this node"), 503
        app._bus.publish("lab/router/command", {"text": cmd, "source": "dashboard", "id": uuid.uuid4().hex[:12]})
        return jsonify(ok=True, sent=cmd)

    return app


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config/dashboard.yaml")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = configlib.load(a.config)
    http = cfg.get("http", {})
    log.info("lab console on :%s (node=%s)", http.get("port", 8099), cfg.get("node_name"))
    create_app(cfg).run(host=http.get("host", "0.0.0.0"), port=int(http.get("port", 8099)), threaded=True)


if __name__ == "__main__":
    main()
