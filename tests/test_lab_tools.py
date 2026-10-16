"""Homelab lab tools: topics, validation and camera calls, with fake transports."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "homelab"))
from lab_tools import Lab, LabError  # noqa: E402


def make_lab():
    sent, urls = [], []

    def fake_get(url, timeout=20):
        urls.append(url)
        return b'{"markers": {"0": [[1, 2]]}}' if "/markers/" in url else b"\xff\xd8jpeg"

    lab = Lab(mqtt_host="orin", camera_url="http://shop-pi:8080/", publish=lambda t, p: sent.append((t, p)),
              http_get=fake_get)
    return lab, sent, urls


def raises(fn, *args):
    try:
        fn(*args)
    except LabError:
        return True
    return False


def test_drawing_and_devices():
    lab, sent, _ = make_lab()
    lab.show_text("hello")
    lab.clear_bench()
    lab.set_device("exhaust", "on")
    assert sent[0][0] == "lab/display/bench" and sent[0][1]["items"][0]["text"] == "hello"
    assert sent[1] == ("lab/display/bench", {"items": [], "append": False})
    assert sent[2] == ("lab/device/exhaust/set", {"state": "on"})


def test_rejects_bad_input():
    lab, sent, _ = make_lab()
    assert raises(lab.set_device, "#", "on")              # MQTT wildcard
    assert raises(lab.set_device, "lights/../x", "on")    # extra topic levels
    assert raises(lab.set_device, "lights", "explode")
    assert raises(lab.show_on_bench, [{"type": "script"}])
    assert raises(lab.show_on_bench, [{"type": "rect"}] * 501)
    assert sent == []


def test_camera_calls():
    lab, _, urls = make_lab()
    assert lab.bench_snapshot(99999).startswith(b"\xff\xd8")
    assert lab.bench_markers()["markers"]["0"] == [[1, 2]]
    assert urls == ["http://shop-pi:8080/snapshot/bench-overhead.jpg?max_width=4096",
                    "http://shop-pi:8080/markers/bench-overhead"]
    assert raises(Lab(camera_url="", publish=lambda *a: None).bench_markers)
