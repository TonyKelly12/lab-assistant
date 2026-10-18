#!/usr/bin/env python3
"""Brain bridge.  NODE: homelab (next to Hermes / the main-brain API).

Closes the loop between the shop bench and the homelab brain. The Orin router
sends anything that needs real knowledge to MQTT `lab/brain/request`; this
bridge answers it with the main-brain API (RAG over datasheets/components) and
publishes `lab/brain/reply`, which the router already knows how to speak and
draw (`router.on_brain_reply`).

    lab/brain/request  {"id","text"}  --> main-brain /chat  --> lab/brain/reply {"id","text","scene"}

Self-contained (standard library + paho-mqtt only) so it can be dropped next to
Hermes without the rest of the repo. All settings come from the environment:

    BRAIN_API_URL     main-brain API base (e.g. http://mainbrain-api:8000)
    BRAIN_API_TOKEN   bearer token (agent token minted with api:read scope)
    BRAIN_USE_DATASHEETS  "1" to turn on the datasheet/component RAG (default on)
    LAB_MQTT_HOST / LAB_MQTT_PORT / LAB_MQTT_USER / LAB_MQTT_PASS   the Orin broker
    BRAIN_DRAW        "1" to also draw a short headline on the bench (default on)

Run:  python3 homelab/brain_bridge.py
"""
import json
import logging
import os
import textwrap
import urllib.request

log = logging.getLogger("brain-bridge")

API_URL = os.environ.get("BRAIN_API_URL", "http://mainbrain-api:8000").rstrip("/")
API_TOKEN = os.environ.get("BRAIN_API_TOKEN", "").strip()
USE_DATASHEETS = os.environ.get("BRAIN_USE_DATASHEETS", "1") not in ("0", "false", "no")
DRAW = os.environ.get("BRAIN_DRAW", "1") not in ("0", "false", "no")
SPEAK_MAX = int(os.environ.get("BRAIN_SPEAK_MAX", "400"))


def ask_brain(text):
    """POST the question to the main-brain /chat endpoint; return the answer text."""
    body = json.dumps({
        "message": text,
        "include_context": True,
        "use_datasheets": USE_DATASHEETS,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"
    req = urllib.request.Request(f"{API_URL}/chat", data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    return (data.get("response") or "").strip()


def bench_headline(text, max_lines=4, width=28):
    """A few short lines for the bench projector (the full answer is spoken)."""
    first = text.strip().split("\n", 1)[0]
    lines = textwrap.wrap(first, width=width)[:max_lines] or ["(no answer)"]
    items, y = [], 60.0
    for line in lines:
        items.append({"type": "text", "x": 100.0, "y": y, "text": line,
                      "size": 8.0, "color": "#00e5ff"})
        y += 26.0
    return {"items": items}


def handle(text):
    """Return (speak_text, scene_or_None) for one request."""
    try:
        answer = ask_brain(text)
    except Exception as exc:  # noqa: BLE001
        log.warning("brain call failed: %s", exc)
        return "I couldn't reach the main brain just now.", None
    if not answer:
        return "I don't have an answer for that.", None
    spoken = answer if len(answer) <= SPEAK_MAX else answer[:SPEAK_MAX].rsplit(" ", 1)[0] + "…"
    scene = bench_headline(answer) if DRAW else None
    return spoken, scene


def _client():
    import paho.mqtt.client as mqtt
    try:
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="brain-bridge")
    except AttributeError:  # paho 1.x
        c = mqtt.Client(client_id="brain-bridge")
    user = os.environ.get("LAB_MQTT_USER")
    if user:
        c.username_pw_set(user, os.environ.get("LAB_MQTT_PASS"))
    return c


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    host = os.environ.get("LAB_MQTT_HOST", "127.0.0.1")
    port = int(os.environ.get("LAB_MQTT_PORT", "1883"))
    client = _client()

    def on_connect(c, *a):
        c.subscribe("lab/brain/request")
        log.info("connected; listening on lab/brain/request")

    def on_message(c, userdata, msg):
        try:
            payload = json.loads(msg.payload or b"null")
        except ValueError:
            payload = {"text": msg.payload.decode("utf-8", "replace")}
        if not isinstance(payload, dict):
            return
        text, rid = str(payload.get("text", "")).strip(), payload.get("id")
        if not text:
            return
        log.info("request %s: %r", rid, text[:80])
        spoken, scene = handle(text)
        reply = {"id": rid, "text": spoken}
        if scene:
            reply["scene"] = scene
        c.publish("lab/brain/reply", json.dumps(reply))

    client.on_connect, client.on_message = on_connect, on_message
    client.connect(host, port, keepalive=30)
    client.loop_forever()


if __name__ == "__main__":
    main()
