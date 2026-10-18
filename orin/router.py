#!/usr/bin/env python3
"""Intent router: turns a sentence into a lab action. Runs on the Orin.

    "turn on the exhaust fan"   -> lab/device/exhaust/set   {"state": "on"}
    "show me a breadboard grid" -> lab/display/bench        {"items": [grid]}
    "why won't my ESP32 boot?"  -> lab/brain/request        (main brain answers)

How a sentence is handled:
  1. Simple rules catch the common commands instantly (clear, on/off, grid).
  2. Anything else goes to the small local model (Ollama), which must answer
     in a fixed JSON shape, so its reply can always be checked.
  3. Questions that need knowledge or reasoning are passed up to the main
     brain on lab/brain/request; its answer comes back on lab/brain/reply.

Every decision is logged to a JSONL file (one line each) along with any
feedback, so there is data to fine-tune on later if the model keeps
getting your phrasing wrong.

MQTT topics
  in   lab/router/command     {"text": "...", "source": "voice", "id": "optional"}  (or plain text)
  in   lab/router/feedback    {"id": "...", "correct": false, "expected": "..."}
  in   lab/brain/reply        {"id": "...", "text": "...", "scene": {...}?, "surface": "bench"?}
  out  lab/router/reply       {"id": "...", "text": "Exhaust fan on.", "intent": "set_device"}
  out  lab/display/<surface>, lab/device/<name>/set, lab/brain/request

Run as a service:   python3 orin/router.py --config config/orin.yaml
Try a sentence:     python3 orin/router.py --config config/orin.yaml try "turn on the lights"
                    (prints the decision, sends nothing)
Mark one wrong:     python3 orin/router.py --config config/orin.yaml feedback a1b2c3d4 --wrong \
                        --expected "set_device lights on"
"""
import argparse
import json
import logging
import os
import re
import signal
import sys
import threading
import time
import urllib.request
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labassist import config as configlib  # noqa: E402
from labassist.mqtt import Bus  # noqa: E402

log = logging.getLogger("router")

INTENTS = ["show_text", "show_grid", "clear_bench", "set_device", "highlight_board", "ask_brain"]


# ------------------------------------------------------------------ settings

DEFAULTS = {
    "surfaces": {
        # Drawable area per projector surface, bench millimetres [x0, y0, x1, y1].
        "bench": {"area_mm": [0, 0, 1200, 600], "grid_origin_mm": [100, 100]},
    },
    "default_surface": "bench",
    "grid_default": [30, 10],
    # Device name -> other names you might say for it.
    "devices": {
        "lights": ["light", "bench lights", "shop lights", "lamp"],
        "exhaust": ["exhaust fan", "fume extractor", "extractor", "fume fan"],
        "fan": ["desk fan", "bench fan"],
    },
    "ollama": {"url": "http://127.0.0.1:11434", "model": "qwen2.5:3b", "timeout_s": 30},
    "log_file": "logs/router.jsonl",
    "wake_words": ["jarvis", "hey jarvis", "ok jarvis"],
}


def settings(cfg):
    s = json.loads(json.dumps(DEFAULTS))
    for key, value in (cfg or {}).items():
        if isinstance(value, dict) and isinstance(s.get(key), dict) and key != "devices" and key != "surfaces":
            s[key].update(value)
        else:
            s[key] = value
    return s


# ------------------------------------------------------------------ text tidy-up

def normalise(text, wake_words=()):
    t = text.lower().strip()
    t = re.sub(r"[^\w\s'\-./]", " ", t)          # drop punctuation except a few
    t = re.sub(r"\s+", " ", t).strip(" .")      # keep dots inside words (v1.2), not at the end
    for w in sorted(wake_words, key=len, reverse=True):
        if t.startswith(w + " "):
            t = t[len(w) + 1:]
            break
    t = re.sub(r"^(please|can you|could you|would you)\s+", "", t)
    t = re.sub(r"\s+please$", "", t)
    return t.strip()


def match_device(name, devices):
    """Map what was said ('the exhaust fan') to a configured device name, or None."""
    name = re.sub(r"^(the|my)\s+", "", name.strip())
    for device, aliases in devices.items():
        if name == device or name in aliases:
            return device
    return None


# ------------------------------------------------------------------ step 1: rules

_ON_OFF = [
    re.compile(r"^(?:turn|switch|shut|put)\s+(on|off)\s+(.+)$"),
    re.compile(r"^(?:turn|switch|shut|put)\s+(.+?)\s+(on|off)$"),
    re.compile(r"^(.+?)\s+(on|off)$"),
]
_CLEAR = re.compile(r"^(?:clear|wipe|reset|blank)(?:\s+(?:the|it|everything|all))?"
                    r"(?:\s+(?:bench|projection|projector|display|screen))?$")
_GRID = re.compile(r"^(?:show|display|project|draw|give)(?:\s+me)?(?:\s+(?:a|the))?"
                   r"(?:\s+(?:breadboard|hole|holes))?\s+grid"
                   r"(?:\s+(\d+)\s*(?:by|x)\s*(\d+))?$")
_TEXT = re.compile(r"^(?:write|show text|show the text|display text|project text|show the words)\s+(.+)$")

# Board-aware projection cues. When one matches, the whole (normalised) sentence
# is forwarded to the bench-vision service as a free-text `ref`; it owns the
# parsing (column/row/hole/rail/place/find/clear) since it has the live board.
_BOARD_CUES = [
    re.compile(r"\b(?:bread)?board\b.*\b(find|scan|outline|clear|wipe|where|locate|show)\b"),
    re.compile(r"\b(find|scan|outline|locate|where\b.*\bis)\b.*\b(?:bread)?board\b"),
    re.compile(r"\b(?:column|col)\s*#?\s*\d"),
    re.compile(r"\brow\s*[a-j]\b"),
    re.compile(r"\bhole\s*[a-j]\s*\d"),
    re.compile(r"\b(?:power|ground|positive|negative)?\s*rails?\b"),
    re.compile(r"\b(?:place|put|mount|seat)\b.*\b(?:column|col)\s*\d"),
]


def board_ref(text):
    """Return the ref to forward to bench-vision if this looks like a board command."""
    return text if any(p.search(text) for p in _BOARD_CUES) else None


def rule_decision(text, s):
    m = _CLEAR.match(text)
    if m:
        return {"intent": "clear_bench"}
    m = _GRID.match(text)
    if m:
        d = {"intent": "show_grid"}
        if m.group(1):
            d["cols"], d["rows"] = int(m.group(1)), int(m.group(2))
        return d
    m = _TEXT.match(text)
    if m:
        return {"intent": "show_text", "text": m.group(1).strip("'\" ")}
    ref = board_ref(text)
    if ref:
        return {"intent": "highlight_board", "ref": ref}
    for i, pattern in enumerate(_ON_OFF):
        m = pattern.match(text)
        if not m:
            continue
        state, name = (m.group(1), m.group(2)) if i == 0 else (m.group(2), m.group(1))
        name = re.sub(r"^(?:turn|switch|shut|put)\s+", "", name)
        device = match_device(name, s["devices"])
        if device:
            return {"intent": "set_device", "device": device, "state": state}
    return None


# ------------------------------------------------------------------ step 2: local model

def build_schema(s):
    props = {
        "intent": {"type": "string", "enum": INTENTS},
        "text": {"type": "string"},
        "ref": {"type": "string"},
        "state": {"type": "string", "enum": ["on", "off"]},
        "cols": {"type": "integer"},
        "rows": {"type": "integer"},
        "surface": {"type": "string", "enum": list(s["surfaces"])},
    }
    if s["devices"]:
        props["device"] = {"type": "string", "enum": list(s["devices"])}
    return {"type": "object", "properties": props, "required": ["intent"]}


def build_prompt(s):
    devices = ", ".join(f"{d} (also: {', '.join(a)})" if a else d for d, a in s["devices"].items()) or "none"
    return f"""You route spoken commands in an electronics workshop to one action.
Reply with JSON only, using exactly one intent:

- show_text: put short text on the bench projector. Fields: text.
- show_grid: show a breadboard hole grid. Fields: cols, rows (only if the user gave them).
- clear_bench: remove everything from the projector.
- set_device: switch a device. Fields: device, state ("on" or "off").
  Devices: {devices}.
- highlight_board: anything about the breadboard on the bench - finding it, or
  lighting up a column, a row, a hole, the power rails, or where a part goes.
  Fields: ref (the request, unchanged - the bench service parses it).
- ask_brain: anything else - questions, explanations, datasheets, pinouts,
  debugging help, or anything you are unsure about. Fields: text (the request, unchanged).

Never invent a device that is not in the list; use ask_brain instead.

Examples:
"put the word solder here on the bench" -> {{"intent": "show_text", "text": "solder"}}
"get rid of the fumes" -> {{"intent": "set_device", "device": "exhaust", "state": "on"}}
"I need a twenty by five hole grid" -> {{"intent": "show_grid", "cols": 20, "rows": 5}}
"I'm done, wipe it all" -> {{"intent": "clear_bench"}}
"find my breadboard" -> {{"intent": "highlight_board", "ref": "find my breadboard"}}
"light up column 12" -> {{"intent": "highlight_board", "ref": "light up column 12"}}
"highlight the power rails" -> {{"intent": "highlight_board", "ref": "highlight the power rails"}}
"place the ESP32 at column 10" -> {{"intent": "highlight_board", "ref": "place the ESP32 at column 10"}}
"what's the pinout of an ESP32-S3" -> {{"intent": "ask_brain", "text": "what's the pinout of an ESP32-S3"}}
"why is my LED so dim" -> {{"intent": "ask_brain", "text": "why is my LED so dim"}}"""


def ollama_decision(text, s, post=None):
    """Ask the local model. Returns (decision dict, raw reply text)."""
    o = s["ollama"]
    body = {
        "model": o["model"],
        "stream": False,
        "format": build_schema(s),
        "options": {"temperature": 0},
        "messages": [{"role": "system", "content": build_prompt(s)},
                     {"role": "user", "content": text}],
    }
    post = post or _post_json
    reply = post(o["url"].rstrip("/") + "/api/chat", body, o.get("timeout_s", 30))
    raw = reply["message"]["content"]
    return json.loads(raw), raw


def _post_json(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def clean(decision, original_text, s):
    """Check a decision and fill in defaults. Anything doubtful becomes ask_brain."""
    d = {k: v for k, v in (decision or {}).items() if v not in (None, "")}
    intent = d.get("intent")
    out = {"intent": intent}
    if d.get("surface") in s["surfaces"]:
        out["surface"] = d["surface"]
    if intent == "set_device":
        device = d.get("device")
        if device not in s["devices"]:
            device = match_device(str(device or ""), s["devices"])
        if device and d.get("state") in ("on", "off"):
            out.update(device=device, state=d["state"])
            return out
    elif intent == "show_text" and str(d.get("text", "")).strip():
        out["text"] = str(d["text"]).strip()[:80]
        return out
    elif intent == "highlight_board" and str(d.get("ref", "")).strip():
        out["ref"] = str(d["ref"]).strip()[:120]
        return out
    elif intent == "show_grid":
        cols, rows = s["grid_default"]
        try:
            cols = int(d.get("cols", cols))
            rows = int(d.get("rows", rows))
        except (TypeError, ValueError):
            pass
        out.update(cols=max(1, min(cols, 64)), rows=max(1, min(rows, 64)))
        return out
    elif intent == "clear_bench":
        return out
    return {"intent": "ask_brain", "text": original_text}


# ------------------------------------------------------------------ actions

def actions_for(decision, request_id, original_text, source, s):
    """Turn a decision into (MQTT messages to send, short reply to speak)."""
    intent = decision["intent"]
    surface = decision.get("surface", s["default_surface"])
    area = s["surfaces"].get(surface, s["surfaces"][s["default_surface"]])["area_mm"]
    display = f"lab/display/{surface}"
    if intent == "clear_bench":
        return [(display, {"items": []})], "Cleared."
    if intent == "show_text":
        cx, cy = (area[0] + area[2]) / 2, (area[1] + area[3]) / 2
        item = {"type": "text", "x": cx, "y": cy, "text": decision["text"], "size": 24,
                "anchor": "middle", "color": "#00e5ff"}
        return [(display, {"items": [item]})], "Done."
    if intent == "show_grid":
        cols, rows = decision["cols"], decision["rows"]
        ox, oy = s["surfaces"].get(surface, {}).get("grid_origin_mm", [area[0] + 50, area[1] + 50])
        pitch = 2.54
        item = {"type": "grid", "x": ox, "y": oy, "w": (cols - 1) * pitch, "h": (rows - 1) * pitch,
                "pitch": pitch, "dots": True, "dot": 0.6, "stroke": "#00e5ff"}
        return [(display, {"items": [item]})], f"{cols} by {rows} grid."
    if intent == "set_device":
        return ([(f"lab/device/{decision['device']}/set", {"state": decision["state"]})],
                f"{decision['device'].capitalize()} {decision['state']}.")
    if intent == "highlight_board":
        # bench-vision does the drawing and sends the spoken reply (same id),
        # so the router stays quiet here (empty reply -> not published).
        return [("lab/board/request", {"ref": decision.get("ref", original_text), "id": request_id})], ""
    # ask_brain
    return ([("lab/brain/request", {"id": request_id, "text": decision.get("text", original_text),
                                    "source": source})],
            "Let me think about that.")


# ------------------------------------------------------------------ the router

class DecisionLog:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

    def write(self, record):
        record = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
        with self.lock, open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


class Router:
    def __init__(self, s, publish, post=None):
        self.s = s
        self.publish = publish          # function(topic, payload)
        self.post = post                # for tests: fake Ollama
        self.log = DecisionLog(s["log_file"])

    def decide(self, text):
        """Returns (decision, details) without sending anything."""
        start = time.time()
        norm = normalise(text, self.s["wake_words"])
        details = {"normalised": norm}
        decision = rule_decision(norm, self.s)
        if decision is not None:
            details["path"] = "rule"
        else:
            try:
                raw_decision, raw = ollama_decision(norm, self.s, self.post)
                details.update(path="model", model=self.s["ollama"]["model"], model_raw=raw)
                decision = raw_decision
            except Exception as exc:
                log.warning("local model failed (%s); passing to the main brain", exc)
                details.update(path="model_error", error=str(exc))
                decision = {"intent": "ask_brain"}
        decision = clean(decision, text.strip(), self.s)
        details["latency_ms"] = round((time.time() - start) * 1000)
        return decision, details

    def handle(self, text, source="unknown", dry_run=False, request_id=None):
        """request_id lets a caller (e.g. Hermes) match the reply to its question."""
        if not (isinstance(request_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,32}", request_id)):
            request_id = uuid.uuid4().hex[:8]
        decision, details = self.decide(text)
        messages, reply = actions_for(decision, request_id, text.strip(), source, self.s)
        self.log.write({"type": "decision", "id": request_id, "text": text, "source": source,
                        "decision": decision, "dry_run": dry_run, **details})
        if not dry_run:
            for topic, payload in messages:
                self.publish(topic, payload)
            # Empty reply means a downstream service (bench-vision) will send the
            # spoken reply with this same id, so the router doesn't double up.
            if reply:
                self.publish("lab/router/reply", {"id": request_id, "text": reply, "intent": decision["intent"]})
        return {"id": request_id, "decision": decision, "messages": messages, "reply": reply, **details}

    # MQTT handlers -----------------------------------------------------
    def on_command(self, topic, payload):
        request_id = None
        if isinstance(payload, dict):
            text, source = str(payload.get("text", "")), str(payload.get("source", "mqtt"))[:32]
            request_id = payload.get("id")
        else:
            text, source = str(payload or ""), "mqtt"
        if not text.strip():
            return
        result = self.handle(text[:500], source, request_id=request_id)
        log.info("%s %r -> %s (%s, %s ms)", result["id"], text, result["decision"],
                 result["path"], result["latency_ms"])

    def on_feedback(self, topic, payload):
        if isinstance(payload, dict) and payload.get("id"):
            self.log.write({"type": "feedback", **payload})

    def on_brain_reply(self, topic, payload):
        if not isinstance(payload, dict):
            payload = {"text": str(payload)}
        if payload.get("scene"):
            surface = payload.get("surface", self.s["default_surface"])
            self.publish(f"lab/display/{surface}", payload["scene"])
        if payload.get("text"):
            self.publish("lab/router/reply", {"id": payload.get("id"), "text": payload["text"],
                                              "intent": "ask_brain"})
        self.log.write({"type": "brain_reply", "id": payload.get("id"), "text": payload.get("text"),
                        "has_scene": bool(payload.get("scene"))})


# ------------------------------------------------------------------ command line

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config/orin.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd")
    t = sub.add_parser("try", help="show what a sentence would do, without sending anything")
    t.add_argument("text", nargs="+")
    f = sub.add_parser("feedback", help="mark a logged decision right or wrong")
    f.add_argument("id")
    g = f.add_mutually_exclusive_group(required=True)
    g.add_argument("--right", action="store_true")
    g.add_argument("--wrong", action="store_true")
    f.add_argument("--expected", help="what it should have done, in your own words")
    a = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = configlib.load(a.config)
    s = settings(cfg.get("router"))
    mqtt_cfg = dict(cfg.get("mqtt") or {})
    if os.environ.get("MQTT_PASS"):           # keeps the password out of the config file if you like
        mqtt_cfg["password"] = os.environ["MQTT_PASS"]

    if a.cmd == "feedback":
        DecisionLog(s["log_file"]).write({"type": "feedback", "id": a.id, "correct": bool(a.right),
                                          "expected": a.expected, "source": "cli"})
        print("noted")
        return
    if a.cmd == "try":
        result = Router(s, publish=lambda *x: None).handle(" ".join(a.text), source="cli", dry_run=True)
        print(json.dumps({k: result[k] for k in ("id", "path", "latency_ms", "decision", "reply", "messages")},
                         indent=2))
        return

    bus = Bus(mqtt_cfg, client_id=cfg.get("node_name", "orin-router"))
    if bus.client is None:
        sys.exit("MQTT is not enabled or paho-mqtt is missing; check the mqtt section of the config.")
    router = Router(s, publish=bus.publish)
    bus.subscribe("lab/router/command", router.on_command)
    bus.subscribe("lab/router/feedback", router.on_feedback)
    bus.subscribe("lab/brain/reply", router.on_brain_reply)
    log.info("router running; model %s, log %s", s["ollama"]["model"], s["log_file"])

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    stop.wait()


if __name__ == "__main__":
    main()
