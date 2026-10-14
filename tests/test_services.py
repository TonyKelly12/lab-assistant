"""Service tests with fake cameras and Flask's test client (no hardware, no network)."""
import json
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera_node.service import Camera, create_app as camera_app  # noqa: E402
from labassist import aruco  # noqa: E402
from labassist.mqtt import Bus  # noqa: E402
from render_node.server import State, create_app as render_app  # noqa: E402


def test_camera_node_snapshot_and_markers():
    img = np.full((1944, 2592, 3), 180, np.uint8)
    img[500:800, 600:900] = cv2.cvtColor(aruco.marker_image(2, 300), cv2.COLOR_GRAY2BGR)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        cv2.imwrite(f.name, img)
    cam = Camera("test", {"type": "fake", "image": f.name, "track": False}, Bus(None, "t"))
    client = camera_app({"test": cam}).test_client()

    home = client.get("/")
    assert home.status_code == 200 and b"/snapshot/test.jpg" in home.data

    r = client.get("/snapshot/test.jpg?max_width=1000")
    assert r.status_code == 200 and r.mimetype == "image/jpeg"
    decoded = cv2.imdecode(np.frombuffer(r.data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[1] == 1000

    markers = client.get("/markers/test").get_json()["markers"]
    assert "2" in markers
    assert client.get("/snapshot/nope.jpg").status_code == 404


def test_render_node_flow():
    with tempfile.TemporaryDirectory() as d:
        state = State("bench", [1920, 1080], os.path.join(d, "bench.json"))
        client = render_app(state).test_client()
        assert client.get("/").status_code == 200
        assert client.get("/static/render.js").status_code == 200
        assert client.post("/pattern", json={"kind": "white"}).get_json()["ok"]
        assert state.mode == "pattern"
        ok, png = cv2.imencode(".png", np.zeros((1080, 1920), np.uint8))
        client.post("/pattern", data=png.tobytes(), content_type="image/png")
        assert state.pattern["kind"] == "image"
        assert client.get(state.pattern["url"]).status_code == 200
        from labassist.calib import focus_pattern
        chart = focus_pattern(1920, 1080)
        assert chart.shape == (1080, 1920) and chart.max() == 255
        ok, png = cv2.imencode(".png", chart)
        assert client.post("/pattern", data=png.tobytes(), content_type="image/png").get_json()["ok"]
        cal = {"H_bench_to_proj": np.eye(3).tolist(), "area_mm": [-60, -60, 960, 510]}
        assert client.post("/calibration", json=cal).get_json()["ok"]
        assert os.path.exists(os.path.join(d, "bench.json"))
        client.post("/scene", json={"items": [{"type": "rect", "x": 1, "y": 2, "w": 3, "h": 4}]})
        s = client.get("/state").get_json()
        assert s["mode"] == "scene" and len(s["scene"]["items"]) == 1
        client.post("/scene", json={"items": [{"type": "cross", "x": 5, "y": 5}], "append": True})
        assert len(json.loads(json.dumps(state.scene))["items"]) == 2


if __name__ == "__main__":
    test_camera_node_snapshot_and_markers()
    test_render_node_flow()
    print("all tests passed")
