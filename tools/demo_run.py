#!/usr/bin/env python3
"""Manual demo driver -- the backup when voice isn't cooperating.

Fires each storyboard beat over MQTT on exactly the path the voice stack uses
(`lab/router/command {text, source:"demo", id}`), so the bench behaves the same
as if you'd spoken it -- you just trigger the beats by hand and dub the voice in
editing. Prints the on-camera line to read and whatever the system says back.

Runs from any machine on the broker's network. It does NOT need the voice
services; the router, bench-vision and brain bridge do the work.

  python3 tools/demo_run.py --config config/orin.yaml              # step through: Enter fires each beat
  python3 tools/demo_run.py --config config/orin.yaml --auto --delay 6
  python3 tools/demo_run.py --config config/orin.yaml --list       # print the script, send nothing
  python3 tools/demo_run.py --config config/orin.yaml --from 3     # start at beat 3
  python3 tools/demo_run.py --config config/orin.yaml --script config/demo_script.yaml

The password comes from the config's mqtt block or the MQTT_PASS env var.
"""
import argparse
import json
import os
import sys
import threading
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labassist import config as configlib  # noqa: E402

# Default storyboard (see docs/DEMO.md). `say` is the line to read on camera;
# `cmd` is what gets sent to the router. A step with `wait` pauses after firing.
DEFAULT_SCRIPT = [
    {"beat": "Cold open", "say": "Jarvis, turn on the bench lights.", "cmd": "turn on the bench lights"},
    {"beat": "Cold open", "say": "Kill the fumes for me.", "cmd": "turn on the exhaust fan"},
    {"beat": "The reveal", "say": "Jarvis, find my board.", "cmd": "find my board"},
    {"beat": "The hero", "say": "Light up column 12.", "cmd": "light up column 12"},
    {"beat": "The hero", "say": "Highlight the power rails.", "cmd": "highlight the power rails"},
    {"beat": "The hero", "say": "Show me where the ESP32 goes.", "cmd": "place the ESP32 at column 10"},
    {"beat": "The brain", "say": "Why won't my ESP32 boot from USB?",
     "cmd": "why won't my ESP32 boot from USB?", "wait": 12},
    {"beat": "Reset", "say": "Clear the board.", "cmd": "clear the board"},
]


def load_script(path):
    if not path:
        return DEFAULT_SCRIPT
    data = configlib.load(path)
    steps = data.get("steps") if isinstance(data, dict) else data
    if not steps:
        sys.exit(f"no steps found in {path}")
    return steps


def make_client(mqtt_cfg):
    import paho.mqtt.client as mqtt
    try:
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"demo-run-{uuid.uuid4().hex[:6]}")
    except AttributeError:  # paho 1.x
        c = mqtt.Client(client_id=f"demo-run-{uuid.uuid4().hex[:6]}")
    user = mqtt_cfg.get("username")
    if user:
        c.username_pw_set(user, os.environ.get("MQTT_PASS") or mqtt_cfg.get("password"))
    return c


class Demo:
    def __init__(self, mqtt_cfg):
        self.replies = []          # (id, text) seen on lab/router/reply
        self.lock = threading.Lock()
        self.client = make_client(mqtt_cfg)
        self.client.on_message = self._on_message
        self.client.connect(mqtt_cfg.get("host", "localhost"), int(mqtt_cfg.get("port", 1883)), keepalive=30)
        self.client.subscribe("lab/router/reply")
        self.client.loop_start()

    def _on_message(self, c, u, msg):
        try:
            d = json.loads(msg.payload or b"null")
        except ValueError:
            return
        if isinstance(d, dict) and d.get("text"):
            with self.lock:
                self.replies.append((d.get("id"), d["text"]))

    def fire(self, cmd, listen=4.0):
        rid = uuid.uuid4().hex[:12]
        with self.lock:
            self.replies.clear()
        self.client.publish("lab/router/command", json.dumps({"text": cmd, "source": "demo", "id": rid}))
        # collect any spoken replies for this id for a short window
        deadline = time.time() + listen
        seen = []
        while time.time() < deadline:
            with self.lock:
                for r in self.replies:
                    if r[0] in (rid, None) and r not in seen:
                        seen.append(r)
            time.sleep(0.2)
        return [t for _, t in seen]

    def close(self):
        self.client.loop_stop()
        self.client.disconnect()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config/orin.yaml")
    p.add_argument("--script", help="YAML with a 'steps' list (defaults to the built-in storyboard)")
    p.add_argument("--auto", action="store_true", help="advance automatically instead of waiting for Enter")
    p.add_argument("--delay", type=float, default=6.0, help="seconds between beats in --auto mode")
    p.add_argument("--from", dest="start", type=int, default=1, help="start at this beat number")
    p.add_argument("--list", action="store_true", help="print the script and exit (sends nothing)")
    a = p.parse_args()

    steps = load_script(a.script)

    if a.list:
        for i, s in enumerate(steps, 1):
            print(f"{i:2d}. [{s.get('beat','')}]  say: {s.get('say','')!r}  ->  send: {s.get('cmd','')!r}")
        return

    cfg = configlib.load(a.config)
    mqtt_cfg = dict(cfg.get("mqtt") or {})
    if not mqtt_cfg:
        sys.exit("no mqtt section in the config; this script needs the broker settings")
    demo = Demo(mqtt_cfg)
    print(f"Connected to {mqtt_cfg.get('host')}:{mqtt_cfg.get('port', 1883)}. "
          f"{'Auto' if a.auto else 'Press Enter to fire each beat.'}\n")
    try:
        for i, s in enumerate(steps, 1):
            if i < a.start:
                continue
            header = f"[{i}/{len(steps)}] {s.get('beat','')}"
            print(f"\n=== {header} ===")
            print(f"  SAY : {s.get('say','')}")
            print(f"  SEND: {s.get('cmd','')}")
            if a.auto:
                time.sleep(max(0.0, a.delay))
            else:
                try:
                    input("  >>> Enter to fire (Ctrl+C to stop) ")
                except EOFError:
                    break
            replies = demo.fire(s["cmd"])
            for r in replies:
                print(f"  SPOKEN: {r}")
            if not replies:
                print("  (no spoken reply -- fine for device/scene-only beats)")
            if s.get("wait"):
                time.sleep(float(s["wait"]))
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        demo.close()


if __name__ == "__main__":
    main()
