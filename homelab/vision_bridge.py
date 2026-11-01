#!/usr/bin/env python3
"""Vision bridge.  NODE: homelab (next to the 3090 / the vision model).

The recognition tier. The Orin router sends a request when you hold a part up to
the inspection webcam and ask what it is; this bridge wakes the 3090 if it's
asleep, grabs the frame from the shop camera node over HTTP, asks a
vision-language model, and publishes the answer on `lab/brain/reply` - which the
router already knows how to speak (and draw).

    lab/vision/request {"id","mode","camera"}  -->  3090 VLM  -->  lab/brain/reply {"id","text","scene"?}

`mode` is one of:
    general   - "what's this part?"  name the type + value/part + package
    markings  - read the printed text off a chip/module, then say the part
    resistor  - read the colour bands -> value (+ a classical-CV cross-check)

This recognises a part it is *shown*. It does not see parts on the breadboard,
locate them, or verify a build - that stays separate from the geometry side
(`lab/board/*`), on purpose.

Self-contained except the resistor cross-check, which lazily imports
`labassist.resistor` (OpenCV) only in resistor mode; the rest runs on the
standard library + paho-mqtt. All settings come from the environment:

    VISION_OLLAMA_URL   VLM host, Ollama API    (default http://127.0.0.1:11434)
    VISION_MODEL        vision model tag        (default qwen2.5vl:7b)
    VISION_CAMERA_URL   shop camera node, e.g. http://<shop-pi-ip>:8080   (required)
    VISION_CAMERA       camera name             (default inspect)
    VISION_MAX_WIDTH    snapshot width px       (default 1280)
    VISION_DRAW         "1" to also draw a short bench headline (default on)
    VISION_SPEAK_MAX    max spoken characters   (default 400)
    VISION_WAKE         "1" to Wake-on-LAN the 3090 first (default off: always-on)
    VISION_WOL_MAC      the 3090's NIC MAC, e.g. a1:b2:c3:d4:e5:f6
    VISION_WOL_BROADCAST  broadcast address     (default 255.255.255.255)
    VISION_WAKE_TIMEOUT seconds to wait for the model host (default 120)
    LAB_MQTT_HOST / LAB_MQTT_PORT / LAB_MQTT_USER / LAB_MQTT_PASS   the Orin broker

Run:  python3 homelab/vision_bridge.py
"""
import base64
import json
import logging
import os
import socket
import textwrap
import time
import urllib.request

log = logging.getLogger("vision-bridge")

OLLAMA_URL = os.environ.get("VISION_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
MODEL = os.environ.get("VISION_MODEL", "qwen2.5vl:7b")
CAMERA_URL = os.environ.get("VISION_CAMERA_URL", "").rstrip("/")
CAMERA = os.environ.get("VISION_CAMERA", "inspect")
MAX_WIDTH = int(os.environ.get("VISION_MAX_WIDTH", "1280"))
DRAW = os.environ.get("VISION_DRAW", "1") not in ("0", "false", "no")
SPEAK_MAX = int(os.environ.get("VISION_SPEAK_MAX", "400"))
WAKE = os.environ.get("VISION_WAKE", "0") not in ("0", "false", "no")
WOL_MAC = os.environ.get("VISION_WOL_MAC", "").strip()
WOL_BROADCAST = os.environ.get("VISION_WOL_BROADCAST", "255.255.255.255").strip()
WAKE_TIMEOUT = int(os.environ.get("VISION_WAKE_TIMEOUT", "120"))

PROMPTS = {
    "general": (
        "You are an electronics bench assistant. A single electronic component is "
        "held up to the camera. Identify it: give the component type and, if you can "
        "read them, its value or part number and package/footprint. If you are not "
        "sure, say so plainly. Answer in one or two short sentences, as if spoken to "
        "someone standing at a workbench."
    ),
    "markings": (
        "Read the text printed on the chip or module in this image, exactly, line by "
        "line. Then state the single most likely part it identifies and what it does, "
        "in one short sentence. If some text is unreadable, say what you can make out."
    ),
    "resistor": (
        "This image shows a single through-hole resistor. Read its colour bands and "
        "give the resistance value and tolerance. Answer in one short sentence."
    ),
}


# ----------------------------------------------------------------- waking the 3090

def _magic_packet(mac):
    mac_bytes = bytes.fromhex(mac.replace(":", "").replace("-", ""))
    if len(mac_bytes) != 6:
        raise ValueError(f"bad MAC {mac!r}")
    return b"\xff" * 6 + mac_bytes * 16


def _host_up(url, timeout=3):
    """True if the Ollama API answers (model host is awake and serving)."""
    try:
        urllib.request.urlopen(f"{url}/api/tags", timeout=timeout).read()
        return True
    except Exception:  # noqa: BLE001
        return False


def wake_host():
    """Wake-on-LAN the model host and wait for the API, if VISION_WAKE is set.

    No-op (returns True) when waking is disabled or the host is already up.
    Returns False if it never came up within the timeout.
    """
    if not WAKE or not WOL_MAC:
        return True
    if _host_up(OLLAMA_URL):
        return True
    try:
        packet = _magic_packet(WOL_MAC)
    except ValueError as exc:
        log.warning("not sending WoL: %s", exc)
        return _host_up(OLLAMA_URL)
    log.info("waking the vision host (%s)", WOL_MAC)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        for _ in range(3):
            s.sendto(packet, (WOL_BROADCAST, 9))
    deadline = time.time() + WAKE_TIMEOUT
    while time.time() < deadline:
        if _host_up(OLLAMA_URL):
            log.info("vision host is up")
            return True
        time.sleep(3)
    log.warning("vision host did not come up within %ss", WAKE_TIMEOUT)
    return False


# ----------------------------------------------------------------- the camera

def fetch_image(camera=None, max_width=None):
    """JPEG bytes of the inspection camera from the shop camera node."""
    if not CAMERA_URL:
        raise RuntimeError("VISION_CAMERA_URL is not set")
    cam = camera or CAMERA
    url = f"{CAMERA_URL}/snapshot/{cam}.jpg?max_width={max_width or MAX_WIDTH}"
    with urllib.request.urlopen(url, timeout=20) as r:
        return r.read()


# ----------------------------------------------------------------- the VLM

class OllamaVision:
    """Default VLM client: an Ollama server with a vision model."""

    def __init__(self, url=OLLAMA_URL, model=MODEL, timeout=120):
        self.url, self.model, self.timeout = url, model, timeout

    def caption(self, image_bytes, prompt):
        body = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [base64.b64encode(image_bytes).decode()],
            }],
        }
        req = urllib.request.Request(f"{self.url}/api/chat", method="POST",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        return (data.get("message", {}).get("content") or "").strip()


def _resistor_hint(image_bytes):
    """Classical-CV resistor reading to hand the VLM, or None if unavailable."""
    try:
        import numpy as np
        import cv2
        from labassist import resistor as R
    except Exception as exc:  # noqa: BLE001 - OpenCV/labassist optional
        log.info("resistor cross-check unavailable (%s)", exc)
        return None
    try:
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return R.read_resistor(img).get("hypothesis")
    except Exception as exc:  # noqa: BLE001
        log.warning("resistor read failed: %s", exc)
        return None


# ----------------------------------------------------------------- bench headline

def bench_headline(text, max_lines=4, width=28):
    """A few short cyan lines for the bench projector (full answer is spoken)."""
    first = text.strip().split("\n", 1)[0]
    lines = textwrap.wrap(first, width=width)[:max_lines] or ["(no answer)"]
    items, y = [], 60.0
    for line in lines:
        items.append({"type": "text", "x": 100.0, "y": y, "text": line,
                      "size": 8.0, "color": "#00e5ff"})
        y += 26.0
    return {"items": items}


# ----------------------------------------------------------------- one request

def handle(mode, image_bytes, client, resistor_hint=_resistor_hint):
    """Return (speak_text, scene_or_None) for one vision request.

    `client` is any object with .caption(image_bytes, prompt) -> str, so tests
    can inject a fake VLM. `resistor_hint` is injectable for the same reason.
    """
    mode = mode if mode in PROMPTS else "general"
    prompt = PROMPTS[mode]
    if mode == "resistor" and resistor_hint:
        hint = resistor_hint(image_bytes)
        if hint:
            prompt = prompt + "\n\n" + hint + " If they disagree, trust the image."
    try:
        answer = client.caption(image_bytes, prompt)
    except Exception as exc:  # noqa: BLE001
        log.warning("VLM call failed: %s", exc)
        return "I couldn't reach the vision model just now.", None
    if not answer:
        return "I couldn't make out the part.", None
    spoken = answer if len(answer) <= SPEAK_MAX else answer[:SPEAK_MAX].rsplit(" ", 1)[0] + "…"
    scene = bench_headline(answer) if DRAW else None
    return spoken, scene


# ----------------------------------------------------------------- service

def _client():
    import paho.mqtt.client as mqtt
    try:
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="vision-bridge")
    except AttributeError:  # paho 1.x
        c = mqtt.Client(client_id="vision-bridge")
    user = os.environ.get("LAB_MQTT_USER")
    if user:
        c.username_pw_set(user, os.environ.get("LAB_MQTT_PASS"))
    return c


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if not CAMERA_URL:
        log.warning("VISION_CAMERA_URL is not set; requests will fail until it is")
    host = os.environ.get("LAB_MQTT_HOST", "127.0.0.1")
    port = int(os.environ.get("LAB_MQTT_PORT", "1883"))
    vlm = OllamaVision()
    client = _client()

    def on_connect(c, *a):
        c.subscribe("lab/vision/request")
        log.info("connected; listening on lab/vision/request (model %s)", MODEL)

    def on_message(c, userdata, msg):
        try:
            payload = json.loads(msg.payload or b"null")
        except ValueError:
            payload = None
        if not isinstance(payload, dict):
            return
        rid = payload.get("id")
        mode = str(payload.get("mode", "general"))
        camera = payload.get("camera") or CAMERA
        log.info("request %s: mode=%s camera=%s", rid, mode, camera)
        if not wake_host():
            c.publish("lab/brain/reply", json.dumps(
                {"id": rid, "text": "The vision rig isn't responding - it may be powered off."}))
            return
        try:
            image = fetch_image(camera)
        except Exception as exc:  # noqa: BLE001
            log.warning("snapshot failed: %s", exc)
            c.publish("lab/brain/reply", json.dumps(
                {"id": rid, "text": "I couldn't get a picture from the inspection camera."}))
            return
        spoken, scene = handle(mode, image, vlm)
        reply = {"id": rid, "text": spoken}
        if scene:
            reply["scene"] = scene
        c.publish("lab/brain/reply", json.dumps(reply))
        log.info("request %s -> %r", rid, spoken[:80])

    client.on_connect, client.on_message = on_connect, on_message
    client.connect(host, port, keepalive=30)
    client.loop_forever()


if __name__ == "__main__":
    main()
