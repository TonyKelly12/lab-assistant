"""Tiny HTTP helpers (standard library only)."""
import json
import urllib.request

import cv2
import numpy as np


def get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def post_json(url, payload, timeout=10):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


def post_png(url, image, timeout=10):
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG encode failed")
    req = urllib.request.Request(url, data=buf.tobytes(), method="POST", headers={"Content-Type": "image/png"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


def get_image(url, timeout=20):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        data = np.frombuffer(r.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"could not decode image from {url}")
    return img
