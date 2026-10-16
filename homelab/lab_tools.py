"""Lab tools for the homelab AI (Hermes, or anything else that can call Python).

Plain functions, no MCP dependency, so they can be dropped into an existing
MCP server or called from Node-RED / LangGraph. mcp_server.py wraps them as
an MCP server.

Everything goes through the MQTT broker on the Orin, except camera images,
which come straight from the camera service over HTTP.

Settings come from environment variables (never put the password in code):
  LAB_MQTT_HOST     Orin IP or hostname
  LAB_MQTT_PORT     default 1883
  LAB_MQTT_USER     the broker login for the homelab (see docs: give it its own user + ACL)
  LAB_MQTT_PASS
  LAB_CAMERA_URL    e.g. http://<shop-pi-ip>:8080
  LAB_CAMERA        default bench-overhead

Needs: pip install paho-mqtt
"""
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
import uuid

ITEM_TYPES = {"rect", "circle", "line", "polyline", "polygon", "text", "image", "grid", "cross"}
MAX_ITEMS = 500
_NAME = re.compile(r"^[a-z0-9_-]{1,32}$")   # device/surface names: no MQTT wildcards or slashes


class LabError(ValueError):
    pass


def _check_name(kind, name):
    if not isinstance(name, str) or not _NAME.match(name):
        raise LabError(f"bad {kind} name {name!r}: use lowercase letters, digits, - or _")
    return name


def check_items(items):
    """Validate a scene before it reaches the projector."""
    if not isinstance(items, list):
        raise LabError("items must be a list")
    if len(items) > MAX_ITEMS:
        raise LabError(f"too many items ({len(items)} > {MAX_ITEMS})")
    for i, item in enumerate(items):
        if not isinstance(item, dict) or item.get("type") not in ITEM_TYPES:
            raise LabError(f"item {i}: type must be one of {sorted(ITEM_TYPES)}")
    return items


class Lab:
    def __init__(self, mqtt_host=None, mqtt_port=None, user=None, password=None,
                 camera_url=None, camera=None, publish=None, http_get=None):
        env = os.environ.get
        self.host = mqtt_host or env("LAB_MQTT_HOST", "127.0.0.1")
        self.port = int(mqtt_port or env("LAB_MQTT_PORT", "1883"))
        self.user = user or env("LAB_MQTT_USER")
        self.password = password or env("LAB_MQTT_PASS")
        self.camera_url = (camera_url or env("LAB_CAMERA_URL", "")).rstrip("/")
        self.camera = camera or env("LAB_CAMERA", "bench-overhead")
        self._publish = publish or self._mqtt_publish     # replaceable for tests
        self._http_get = http_get or self._get

    # -- transport ------------------------------------------------------
    def _auth(self):
        return {"username": self.user, "password": self.password} if self.user else None

    def _mqtt_publish(self, topic, payload):
        import paho.mqtt.publish as publish
        publish.single(topic, json.dumps(payload), hostname=self.host, port=self.port,
                       auth=self._auth(), client_id=f"lab-tools-{uuid.uuid4().hex[:6]}")

    @staticmethod
    def _get(url, timeout=20):
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read()

    def _camera(self, path):
        if not self.camera_url:
            raise LabError("LAB_CAMERA_URL is not set")
        return self._http_get(f"{self.camera_url}{path}")

    # -- tools ----------------------------------------------------------
    def show_on_bench(self, items, surface="bench", append=False):
        """Draw shapes on a projector surface. Positions and sizes are in millimetres."""
        self._publish(f"lab/display/{_check_name('surface', surface)}",
                      {"items": check_items(items), "append": bool(append)})
        return f"sent {len(items)} item(s) to {surface}"

    def show_text(self, text, x=600.0, y=300.0, size=24.0, surface="bench"):
        """Put one line of text on the bench (default: the middle)."""
        item = {"type": "text", "x": float(x), "y": float(y), "text": str(text)[:200],
                "size": float(size), "anchor": "middle", "color": "#00e5ff"}
        return self.show_on_bench([item], surface)

    def clear_bench(self, surface="bench"):
        """Remove everything from a projector surface."""
        return self.show_on_bench([], surface)

    def set_device(self, device, state):
        """Switch a shop device (lights, exhaust, fan...) on or off."""
        if state not in ("on", "off"):
            raise LabError("state must be 'on' or 'off'")
        self._publish(f"lab/device/{_check_name('device', device)}/set", {"state": state})
        return f"{device} {state}"

    def bench_snapshot(self, max_width=1280):
        """JPEG bytes of the overhead bench camera."""
        max_width = max(160, min(int(max_width), 4096))
        return self._camera(f"/snapshot/{urllib.parse.quote(self.camera)}.jpg?max_width={max_width}")

    def bench_markers(self):
        """ArUco markers the camera currently sees (ids and corner pixels)."""
        return json.loads(self._camera(f"/markers/{urllib.parse.quote(self.camera)}"))

    def ask_lab(self, text, timeout=15.0):
        """Send a sentence to the Orin's router, as if spoken, and wait for its reply."""
        import paho.mqtt.client as mqtt
        request_id = uuid.uuid4().hex[:12]
        done, answer = threading.Event(), {}
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"lab-tools-{request_id}")
        except AttributeError:  # paho 1.x
            client = mqtt.Client(client_id=f"lab-tools-{request_id}")
        if self.user:
            client.username_pw_set(self.user, self.password)

        def on_connect(c, *args):
            c.subscribe("lab/router/reply")
            c.publish("lab/router/command", json.dumps({"text": str(text)[:500], "source": "homelab",
                                                        "id": request_id}))

        def on_message(c, userdata, msg):
            try:
                data = json.loads(msg.payload)
            except ValueError:
                return
            if data.get("id") == request_id:
                answer.update(data)
                done.set()

        client.on_connect, client.on_message = on_connect, on_message
        client.connect(self.host, self.port, keepalive=30)
        client.loop_start()
        try:
            if not done.wait(timeout):
                return {"id": request_id, "error": f"no reply from the router within {timeout:.0f}s"}
            return answer
        finally:
            client.loop_stop()
            client.disconnect()
