"""Breadboard circuit layouts projected onto the real holes.

A *layout* is a wiring plan — where each component sits and which holes connect.
It is authored knowledge, NOT something the camera detects. The board finder
(`labassist.board.find_boards`) gives each board's real position, and
`labassist.breadboard.hole_xy` turns a (column, row) into an exact bench-mm
coordinate, so a wire is simply a line drawn between two hole coordinates.

Because every detected board lives in the same bench-mm frame, a layout can span
several boards: a hole reference carries an optional ``board`` index, and a wire
between holes on different boards is just a line across the gap (you run the real
jumper — the projector shows you where). The layout, never the camera, decides
what connects to what.

This module is pure geometry + data: no hardware, no MQTT. It turns a layout +
the detected board(s) into projector scene items in the same format the render
node already speaks (see ``labassist.breadboard`` scene builders).
"""
import glob
import json
import os
from dataclasses import replace

from . import breadboard as bb

CIRCUITS_DIR = os.environ.get("LAB_CIRCUITS_DIR") or os.path.join(
    os.path.dirname(__file__), os.pardir, "circuits")

# Net name -> wire colour. Keeps the projected plan readable: power red, ground
# grey, signals amber, data/clock cyan. Overridable per wire with "color".
NET_COLORS = {
    "vcc": "#ff4d4d", "5v": "#ff4d4d", "3v3": "#ff4d4d", "+": "#ff4d4d", "power": "#ff4d4d",
    "gnd": "#8a8f98", "ground": "#8a8f98", "-": "#8a8f98",
    "signal": "#ffb000", "gpio": "#ffb000", "pwm": "#ffb000",
    "data": "#00e5ff", "sda": "#00e5ff", "clk": "#00e5ff", "scl": "#35e08a",
}
DEFAULT_WIRE = "#00e5ff"
PART_COLOR = "#9b8cff"      # component footprints/labels (violet)
PIN_COLOR = "#eaf3f7"


# ---------------------------------------------------------------- hole resolution

def _row_v(profile, row):
    """Short-axis mm offset for a row name (a-j) or a rail name (e.g. 'top+')."""
    if row in profile.rows:
        return profile.rows[row]
    if row in profile.rails:
        return profile.rails[row]
    raise KeyError(f"unknown row/rail {row!r} on profile {profile.name!r}")


def bench_profile(board, name=None):
    """The profile to grid a detected surface with.

    Breadboards physically snap together (dovetail locks) into one larger
    continuous surface, with holes running across the seam at the same 2.54 mm
    pitch. From overhead that's simply a longer rectangle, so we number columns
    *continuously* across the whole thing instead of capping at a single board's
    count. `hole_xy` clamps to the real board edge, so nothing projects off it.

    Row offsets/`col_start` come from the nearest standard profile (half/full);
    those are fine-tuned once against a photo and don't change when you add
    length by snapping boards on.
    """
    base = bb.pick_profile(board, name=name)
    return replace(base, name=f"{base.name}+cont", columns=100_000)


def profiles_for(boards, layout=None, name=None):
    """One continuous profile per detected surface. `name` (or a layout's
    'board' hint) forces half/full row geometry; otherwise it's by size."""
    name = name or (layout or {}).get("board")
    return [bench_profile(b, name=name) for b in boards]


def resolve_hole(boards, profiles, ref):
    """Turn a hole reference into a [x, y] bench-mm coordinate.

    ref: {"col": int, "row": "a".."j", "board": int?}            terminal hole
      or {"col": int, "rail": "top+"|"top-"|..., "board": int?}  power-rail hole
    """
    bi = int(ref.get("board", 0))
    if not 0 <= bi < len(boards):
        raise IndexError(f"board index {bi} not available (found {len(boards)})")
    row = ref.get("row") or ref.get("rail")
    if row is None:
        raise KeyError("hole reference needs a 'row' or 'rail'")
    col = int(ref["col"])
    return bb.hole_xy(boards[bi], profiles[bi], col, _row_v(profiles[bi], row))


# ---------------------------------------------------------------- validation

def validate(layout, boards, profiles=None):
    """Return a list of human-readable problems with a layout against the real
    board(s). Empty list means every hole resolves and sits on a real board."""
    profiles = profiles or profiles_for(boards, layout)
    problems = []
    need = int(layout.get("boards", 1))
    if len(boards) < need:
        problems.append(f"layout expects {need} board(s); {len(boards)} detected")

    def check(ref, where):
        try:
            bi = int(ref.get("board", 0))
            if not 0 <= bi < len(boards):
                problems.append(f"{where}: board index {bi} not available")
                return
            col = int(ref["col"])
            ncols = bb._col_count(boards[bi], profiles[bi])
            if not 1 <= col <= ncols:
                problems.append(f"{where}: column {col} out of range 1..{ncols} on board {bi}")
            row = ref.get("row") or ref.get("rail")
            _row_v(profiles[bi], row)
        except (KeyError, IndexError, ValueError, TypeError) as e:
            problems.append(f"{where}: {e}")

    for comp in layout.get("components", []):
        for pin in comp.get("pins", []):
            check(pin, f"{comp.get('ref', comp.get('part', '?'))} pin {pin.get('name', '')}".strip())
    for i, wire in enumerate(layout.get("wires", [])):
        check(wire["from"], f"wire {i} from")
        check(wire["to"], f"wire {i} to")
    return problems


# ---------------------------------------------------------------- rendering

def _footprint(boards, profiles, comp):
    """Quad outline (polygon points) for a component that spans columns across
    two rows, e.g. an IC or dev board straddling the gap."""
    span = comp.get("span")
    rows = comp.get("rows")
    if not (span and rows):
        return None
    bi = int(comp.get("board", 0))
    c0, c1 = int(span[0]), int(span[1])
    r0, r1 = rows[0], rows[1]
    pad = 1.2
    p00 = resolve_hole(boards, profiles, {"col": c0, "row": r0, "board": bi})
    p10 = resolve_hole(boards, profiles, {"col": c1, "row": r0, "board": bi})
    p11 = resolve_hole(boards, profiles, {"col": c1, "row": r1, "board": bi})
    p01 = resolve_hole(boards, profiles, {"col": c0, "row": r1, "board": bi})
    # inflate slightly so the outline frames the pins rather than clipping them
    return [[round(p00[0] - pad, 1), round(p00[1] - pad, 1)],
            [round(p10[0] + pad, 1), round(p10[1] - pad, 1)],
            [round(p11[0] + pad, 1), round(p11[1] + pad, 1)],
            [round(p01[0] - pad, 1), round(p01[1] + pad, 1)]]


def _wire_color(wire):
    if wire.get("color"):
        return wire["color"]
    return NET_COLORS.get(str(wire.get("net", "")).lower(), DEFAULT_WIRE)


def render_circuit(layout, boards, profiles=None, steps=None):
    """Turn a layout + detected board(s) into projector scene items.

    Components are always drawn. Wires are drawn up to ``steps`` (None = all),
    so a caller can step through a build one connection at a time.

    Returns {"items": [...], "steps": <total wires>, "shown": <wires drawn>,
    "title": str}. Raises on an unresolvable hole — call validate() first.
    """
    if not boards:
        raise ValueError("no boards detected to render onto")
    profiles = profiles or profiles_for(boards, layout)
    items = []

    # components: footprint outline, pin dots, label
    for comp in layout.get("components", []):
        fp = _footprint(boards, profiles, comp)
        if fp:
            items.append({"type": "polygon", "points": fp,
                          "stroke": PART_COLOR, "width": 0.8, "fill": "none"})
        pin_xy = []
        for pin in comp.get("pins", []):
            xy = resolve_hole(boards, profiles, pin)
            pin_xy.append(xy)
            items.append({"type": "circle", "cx": xy[0], "cy": xy[1], "r": 1.8,
                          "stroke": PIN_COLOR, "width": 0.5, "fill": "none"})
        label = comp.get("ref") or comp.get("part")
        if label:
            if fp:
                lx, ly = min(p[0] for p in fp), min(p[1] for p in fp)
            elif pin_xy:
                lx, ly = min(p[0] for p in pin_xy), min(p[1] for p in pin_xy)
            else:
                continue
            items.append({"type": "text", "x": lx, "y": ly - 2.5, "size": 4.5,
                          "color": PART_COLOR, "text": label, "bold": True})

    # wires: a coloured line between two real holes, with end dots
    wires = layout.get("wires", [])
    total = len(wires)
    shown = total if steps is None else max(0, min(int(steps), total))
    for wire in wires[:shown]:
        a = resolve_hole(boards, profiles, wire["from"])
        b = resolve_hole(boards, profiles, wire["to"])
        color = _wire_color(wire)
        items.append({"type": "line", "x1": a[0], "y1": a[1], "x2": b[0], "y2": b[1],
                      "stroke": color, "width": 1.0})
        for xy in (a, b):
            items.append({"type": "circle", "cx": xy[0], "cy": xy[1], "r": 1.0,
                          "stroke": color, "width": 0.4, "fill": color})
        if wire.get("label"):
            mx, my = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
            items.append({"type": "text", "x": round(mx, 1), "y": round(my - 1.5, 1),
                          "size": 3.5, "color": color, "text": wire["label"]})

    return {"items": items, "steps": total, "shown": shown,
            "title": layout.get("title") or layout.get("name") or "circuit"}


# ---------------------------------------------------------------- library

def _load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_circuits(directory=None):
    """[{name, title, board, boards, summary}] for every circuit JSON on disk."""
    directory = directory or CIRCUITS_DIR
    out = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        try:
            d = _load_file(path)
        except (OSError, json.JSONDecodeError):
            continue
        out.append({"name": d.get("name") or os.path.splitext(os.path.basename(path))[0],
                    "title": d.get("title", ""), "board": d.get("board", "auto"),
                    "boards": d.get("boards", 1), "summary": d.get("summary", "")})
    return out


def load_circuit(name, directory=None):
    """Load a circuit by its name/filename (case-insensitive)."""
    directory = directory or CIRCUITS_DIR
    want = str(name).lower().strip()
    for path in glob.glob(os.path.join(directory, "*.json")):
        stem = os.path.splitext(os.path.basename(path))[0].lower()
        if stem == want:
            return _load_file(path)
    for path in glob.glob(os.path.join(directory, "*.json")):
        d = _load_file(path)
        if str(d.get("name", "")).lower() == want:
            return d
    raise FileNotFoundError(f"no circuit named {name!r} in {directory}")


def find_circuit(query, directory=None):
    """Best-effort match of a free-text request to a circuit in the library.

    Scores on word overlap against each circuit's name, title, aliases and
    summary. Returns the layout dict, or None if nothing reasonably matches.
    Deterministic library lookup — the generative path (ask the brain to author
    a layout) lives in the router/bridge, not here.
    """
    directory = directory or CIRCUITS_DIR
    q = set(_words(query))
    if not q:
        return None
    best, best_score = None, 0.0
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        try:
            d = _load_file(path)
        except (OSError, json.JSONDecodeError):
            continue
        hay = " ".join([str(d.get("name", "")), str(d.get("title", "")),
                        " ".join(d.get("aliases", [])), str(d.get("summary", ""))])
        words = set(_words(hay))
        if not words:
            continue
        overlap = len(q & words)
        if not overlap:
            continue
        score = overlap / len(q)
        if score > best_score:
            best, best_score = d, score
    # require at least a couple of matching words, or a single strong one
    return best if best_score >= 0.34 else None


def _words(s):
    out, cur = [], []
    for ch in str(s).lower():
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    stop = {"a", "an", "the", "to", "me", "show", "wire", "up", "how", "on",
            "my", "with", "and", "for", "of", "build", "connect", "circuit"}
    return [w for w in out if w not in stop and len(w) > 1]
