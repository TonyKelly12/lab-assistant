"""Optional MQTT client that works with paho-mqtt 1.x and 2.x.

Every node keeps working (over HTTP) when the broker is missing or down;
MQTT just adds push events once the broker on the Orin exists.
"""
import json
import logging
import threading
import uuid

log = logging.getLogger("mqtt")

try:
    import paho.mqtt.client as paho
except ImportError:  # paho not installed
    paho = None


class Bus:
    def __init__(self, cfg, client_id):
        self.cfg = cfg or {}
        self.client = None
        self.connected = threading.Event()
        self._subs = {}
        if not self.cfg.get("enabled") or paho is None:
            if self.cfg.get("enabled"):
                log.warning("paho-mqtt not installed; MQTT disabled")
            return
        # Make the client id unique per process, so a leftover/duplicate
        # instance can't collide and trigger a broker-eviction reconnect loop.
        uid = f"{client_id}-{uuid.uuid4().hex[:6]}"
        try:
            self.client = paho.Client(paho.CallbackAPIVersion.VERSION2, client_id=uid)
        except AttributeError:  # paho 1.x
            self.client = paho.Client(client_id=uid)
        if self.cfg.get("username"):
            self.client.username_pw_set(self.cfg["username"], self.cfg.get("password"))
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = lambda *a: self.connected.clear()
        self.client.on_message = self._on_message
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)
        self.client.connect_async(self.cfg.get("host", "localhost"), int(self.cfg.get("port", 1883)))
        self.client.loop_start()

    def _on_connect(self, client, *args):
        log.info("connected to MQTT broker")
        self.connected.set()
        for topic in self._subs:
            client.subscribe(topic)

    def _on_message(self, client, userdata, msg):
        for topic, handler in self._subs.items():
            if paho.topic_matches_sub(topic, msg.topic):
                try:
                    try:
                        payload = json.loads(msg.payload or b"null")
                    except ValueError:  # plain text, e.g. mosquitto_pub -m "lights on"
                        payload = msg.payload.decode("utf-8", "replace")
                    handler(msg.topic, payload)
                except Exception:
                    log.exception("handler failed for %s", msg.topic)

    def subscribe(self, topic, handler):
        self._subs[topic] = handler
        if self.client and self.connected.is_set():
            self.client.subscribe(topic)

    def publish(self, topic, payload, retain=False):
        if self.client and self.connected.is_set():
            self.client.publish(topic, json.dumps(payload), qos=0, retain=retain)
        elif self.client:
            log.warning("MQTT not connected; dropped publish to %s", topic)
