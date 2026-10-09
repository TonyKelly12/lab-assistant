"""Send something to a projector. Coordinates are bench millimetres.

  python3 tools/send_scene.py --render-url http://bench-render.local:8090 rect 100 80 200 120
  python3 tools/send_scene.py --render-url ... holes 100 80 30 10      # 30x10 hole grid at 2.54 mm
  python3 tools/send_scene.py --render-url ... text 100 60 "Hello bench"
  python3 tools/send_scene.py --render-url ... file my-scene.json
  python3 tools/send_scene.py --render-url ... clear
Add --append to keep what's already shown.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labassist.net import post_json  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--render-url", required=True)
    p.add_argument("--append", action="store_true")
    p.add_argument("what", choices=["rect", "holes", "text", "file", "clear"])
    p.add_argument("args", nargs="*")
    a = p.parse_args()

    if a.what == "clear":
        items = []
    elif a.what == "rect":
        x, y, w, h = map(float, a.args)
        items = [{"type": "rect", "x": x, "y": y, "w": w, "h": h, "stroke": "#00e5ff", "width": 2}]
    elif a.what == "holes":
        x, y, cols, rows = float(a.args[0]), float(a.args[1]), int(a.args[2]), int(a.args[3])
        pitch = 2.54
        items = [{"type": "grid", "x": x, "y": y, "w": (cols - 1) * pitch, "h": (rows - 1) * pitch,
                  "pitch": pitch, "dots": True, "dot": 0.6, "stroke": "#00e5ff"}]
    elif a.what == "text":
        x, y, text = float(a.args[0]), float(a.args[1]), " ".join(a.args[2:])
        items = [{"type": "text", "x": x, "y": y, "text": text, "size": 12}]
    else:
        with open(a.args[0]) as f:
            items = json.load(f).get("items", [])

    print(post_json(f"{a.render_url.rstrip('/')}/scene", {"items": items, "append": a.append}))


if __name__ == "__main__":
    main()
