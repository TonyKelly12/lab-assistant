"""MCP server exposing the lab tools, for Hermes behind your MCP gateway.

  pip install "mcp" paho-mqtt
  export LAB_MQTT_HOST=<orin-ip> LAB_MQTT_USER=homelab LAB_MQTT_PASS=... LAB_CAMERA_URL=http://<shop-pi-ip>:8080
  python3 homelab/mcp_server.py                     # stdio (default)
  python3 homelab/mcp_server.py --http              # streamable HTTP (SDK default host/port)

If you'd rather add these to your existing MCP server, import Lab from
lab_tools.py and register its methods there instead.

Works with the MCP Python SDK 2.x (MCPServer) and 1.x (FastMCP).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab_tools import Lab  # noqa: E402

try:
    from mcp.server.mcpserver import Image, MCPServer as Server   # SDK 2.x
except ImportError:
    from mcp.server.fastmcp import FastMCP as Server, Image       # SDK 1.x

lab = Lab()
server = Server("lab-assistant", instructions=(
    "Tools for Tony's electronics workbench. The bench projector draws in millimetres "
    "(bench about 1200 x 600 mm, origin at the back-left marker, x right, y toward the front). "
    "Look with bench_snapshot before drawing near parts; keep drawings simple and high-contrast."))


@server.tool()
def show_on_bench(items: list[dict], append: bool = False) -> str:
    """Draw on the bench projector. Items (positions in mm):
    rect{x,y,w,h,stroke?,fill?,width?}, circle{cx,cy,r}, line{x1,y1,x2,y2},
    polyline/polygon{points:[[x,y],...]}, text{x,y,text,size?,color?,anchor?},
    image{href,x,y,w,h}, grid{x,y,w,h,pitch?,dots?}, cross{x,y,size?}.
    Each item needs a "type" field. append=true keeps what is already shown."""
    return lab.show_on_bench(items, "bench", append)


@server.tool()
def show_text(text: str, x: float = 600.0, y: float = 300.0, size: float = 24.0) -> str:
    """Show one line of text on the bench (default: the middle)."""
    return lab.show_text(text, x, y, size)


@server.tool()
def clear_bench() -> str:
    """Remove everything from the bench projector."""
    return lab.clear_bench()


@server.tool()
def set_device(device: str, state: str) -> str:
    """Switch a shop device on or off. device: lights, exhaust or fan. state: on or off."""
    return lab.set_device(device, state)


@server.tool()
def bench_snapshot(max_width: int = 1280) -> Image:
    """Photo of the bench from the overhead camera."""
    return Image(data=lab.bench_snapshot(max_width), format="jpeg")


@server.tool()
def bench_markers() -> dict:
    """ArUco markers the bench camera sees right now (ids and corner pixel positions)."""
    return lab.bench_markers()


@server.tool()
def ask_lab(text: str) -> dict:
    """Send a sentence to the lab's own command router, as if Tony said it, and return its reply."""
    return lab.ask_lab(text)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    a = p.parse_args()
    if a.http:
        server.run(transport="streamable-http")
    else:
        server.run()
