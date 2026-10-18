"""Breadboard hole geometry, in bench millimetres.

Runs anywhere (pure maths, no hardware). Takes a board the finder located
(`labassist.board.find_boards` -> center/size/angle/corners in bench mm) and a
profile describing the hole layout, and turns spoken references ("column 12",
"the power rails", "hole e12", "where does the ESP32 go") into exact hole
coordinates on the real bench, plus projector scene items to light them up.

The finder gives the board's real outline, so we lay the hole lattice along the
board's own axes from one corner -- it works at any rotation, the same property
the projector calibration already relies on. Profile distances are in mm and
are meant to be tuned once against a photo via shopvision's /board endpoint.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

PITCH = 2.54  # 0.1" standard hole pitch
CYAN = "#00e5ff"
AMBER = "#ffb000"


@dataclass
class Profile:
    """Hole layout relative to the detected outline, long axis = columns.

    rows maps a row name to its distance (mm) from the origin long edge
    (the short-axis offset). Columns start `col_start` mm in from the origin
    short edge and step by `pitch`. Everything is tunable per board.
    """
    name: str
    columns: int
    col_start: float
    rows: dict                      # name -> v offset mm (a-j terminal rows)
    rails: dict = field(default_factory=dict)   # name -> v offset mm (power rails)
    pitch: float = PITCH
    long_min_mm: float = 120.0      # boards longer than this auto-pick this profile


# Approximate standard boards. Defaults are close; fine-tune `rows`/`col_start`
# once against a real photo. v offsets are mm from the chosen long edge.
PROFILES = {
    # 830-point full board: 63 columns, rails outboard of the terminal rows.
    "full": Profile(
        name="full", columns=63, col_start=7.0, long_min_mm=120.0,
        rails={"top+": 2.5, "top-": 5.0, "bot-": 50.0, "bot+": 52.5},
        rows={"a": 11.0, "b": 13.5, "c": 16.0, "d": 18.5, "e": 21.0,
              "f": 31.5, "g": 34.0, "h": 36.5, "i": 39.0, "j": 41.5},
    ),
    # 400-point half board: 30 columns, no onboard rails on some, a-j terminals.
    "half": Profile(
        name="half", columns=30, col_start=6.0, long_min_mm=0.0,
        rails={},
        rows={"a": 8.0, "b": 10.5, "c": 13.0, "d": 15.5, "e": 18.0,
              "f": 28.5, "g": 31.0, "h": 33.5, "i": 36.0, "j": 38.5},
    ),
}

TERMINAL_ROWS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]


def pick_profile(board, name=None):
    if name and name in PROFILES:
        return PROFILES[name]
    long_mm = max(board["size"])
    return PROFILES["full"] if long_mm >= PROFILES["full"].long_min_mm else PROFILES["half"]


def frame(board):
    """Origin corner + unit axes (u along columns/long side, v across rows).

    Deterministic: origin is the corner nearest the bench origin, so the same
    physical board always indexes the same way. Returns (origin, u_hat, v_hat,
    length_u, length_v) in bench mm.
    """
    corners = np.asarray(board["corners"], dtype=float)
    o = int(np.argmin(corners.sum(axis=1)))
    cur = corners[o]
    nxt = corners[(o + 1) % 4]
    prv = corners[(o - 1) % 4]
    e_next, e_prev = nxt - cur, prv - cur
    # long edge -> columns axis (u); short edge -> rows axis (v)
    if np.linalg.norm(e_next) >= np.linalg.norm(e_prev):
        u_vec, v_vec = e_next, e_prev
    else:
        u_vec, v_vec = e_prev, e_next
    lu, lv = float(np.linalg.norm(u_vec)), float(np.linalg.norm(v_vec))
    u_hat = u_vec / max(lu, 1e-6)
    v_hat = v_vec / max(lv, 1e-6)
    return cur, u_hat, v_hat, lu, lv


def hole_xy(board, profile, col, row_v):
    """Bench mm for a column index (1-based) at short-axis offset row_v (mm)."""
    origin, u_hat, v_hat, lu, lv = frame(board)
    u = profile.col_start + (col - 1) * profile.pitch
    u = min(max(u, 0.0), lu)
    v = min(max(row_v, 0.0), lv)
    p = origin + u_hat * u + v_hat * v
    return [round(float(p[0]), 1), round(float(p[1]), 1)]


def _col_count(board, profile):
    _, _, _, lu, _ = frame(board)
    fit = int((lu - profile.col_start) / profile.pitch) + 1
    return max(1, min(profile.columns, fit))


# ---------------------------------------------------------------- reference parsing

_COL = re.compile(r"(?:column|col|c)\s*#?\s*(\d{1,2})")
_HOLE = re.compile(r"\b([a-j])\s*(\d{1,2})\b")
_ROW = re.compile(r"\brow\s*([a-j])\b")
_PLACE = re.compile(r"(?:place|put|mount|seat)\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+(?:at|on|across|over)\s+(?:column|col|c)?\s*(\d{1,2})")


def resolve(ref, board, profile=None):
    """Turn a reference string into holes/footprint to light up.

    Returns {"kind", "label", "holes": [[x,y],...], "footprint": [[x,y]*4]?}.
    Unrecognised refs return kind "unknown" with an empty hole list.
    """
    profile = profile or pick_profile(board)
    t = (ref or "").strip().lower()
    ncols = _col_count(board, profile)

    def col_holes(c):
        return [hole_xy(board, profile, c, profile.rows[r]) for r in TERMINAL_ROWS]

    # place a part across the centre gap at a column span
    m = _PLACE.search(t)
    if m:
        part, c = m.group(1).strip(), int(m.group(2))
        span = _PART_SPAN.get(_part_key(part), 6)
        c0, c1 = max(1, c), min(ncols, c + span - 1)
        corners = _footprint(board, profile, c0, c1, "a", "j")
        holes = [hole_xy(board, profile, cc, profile.rows["a"]) for cc in range(c0, c1 + 1)]
        holes += [hole_xy(board, profile, cc, profile.rows["j"]) for cc in range(c0, c1 + 1)]
        return {"kind": "place", "label": f"{part} → cols {c0}-{c1}", "holes": holes, "footprint": corners}

    # single hole e12
    m = _HOLE.search(t)
    if m and (m.group(1) in profile.rows):
        row, c = m.group(1), int(m.group(2))
        if 1 <= c <= ncols:
            return {"kind": "hole", "label": f"{row}{c}", "holes": [hole_xy(board, profile, c, profile.rows[row])]}

    # a whole column (its 10 terminal holes)
    m = _COL.search(t)
    if m:
        c = int(m.group(1))
        if 1 <= c <= ncols:
            return {"kind": "column", "label": f"column {c}", "holes": col_holes(c)}

    # a terminal row across all columns
    m = _ROW.search(t)
    if m and m.group(1) in profile.rows:
        row = m.group(1)
        return {"kind": "row", "label": f"row {row}",
                "holes": [hole_xy(board, profile, c, profile.rows[row]) for c in range(1, ncols + 1)]}

    # power rails
    if "rail" in t or "power" in t or "bus" in t or "+" in t or "-" in t or "ground" in t or "gnd" in t or "positive" in t or "negative" in t:
        want = set(profile.rails)
        if "ground" in t or "negative" in t or "gnd" in t or (" -" in f" {t}"):
            want = {k for k in profile.rails if k.endswith("-")}
        elif "positive" in t or ("+" in t):
            want = {k for k in profile.rails if k.endswith("+")}
        holes = [hole_xy(board, profile, c, v) for name, v in profile.rails.items() if name in want
                 for c in range(1, ncols + 1)]
        label = "power rails" if want == set(profile.rails) else f"rail {','.join(sorted(want))}"
        if holes:
            return {"kind": "rail", "label": label, "holes": holes}

    # centre gap / ravine
    if "gap" in t or "ravine" in t or "centre" in t or "center" in t or "middle" in t:
        _, _, _, _, lv = frame(board)
        mid_v = (profile.rows["e"] + profile.rows["f"]) / 2
        return {"kind": "gap", "label": "centre gap",
                "holes": [hole_xy(board, profile, c, mid_v) for c in range(1, ncols + 1)]}

    return {"kind": "unknown", "label": ref, "holes": []}


# part -> how many columns its footprint spans across the gap
_PART_SPAN = {"esp32": 15, "esp8266": 10, "arduino_nano": 15, "pi_pico": 20, "ic": 4, "dip": 4}


def _part_key(part):
    p = re.sub(r"[^a-z0-9]+", "_", part.lower()).strip("_")
    for k in _PART_SPAN:
        if k in p:
            return k
    return p


def _footprint(board, profile, c0, c1, row0, row1):
    v0, v1 = profile.rows[row0], profile.rows[row1]
    pad = profile.pitch * 0.6
    origin, u_hat, v_hat, lu, lv = frame(board)

    def pt(u, v):
        u = min(max(u, 0.0), lu)
        v = min(max(v, 0.0), lv)
        p = origin + u_hat * u + v_hat * v
        return [round(float(p[0]), 1), round(float(p[1]), 1)]

    u0 = profile.col_start + (c0 - 1) * profile.pitch - pad
    u1 = profile.col_start + (c1 - 1) * profile.pitch + pad
    return [pt(u0, v0 - pad), pt(u1, v0 - pad), pt(u1, v1 + pad), pt(u0, v1 + pad)]


# ---------------------------------------------------------------- scene builders

def holes_scene(resolved, color=CYAN, board=None):
    """Scene items: a dot per hole, an optional footprint outline, and a label."""
    items = []
    for x, y in resolved.get("holes", []):
        items.append({"type": "circle", "cx": x, "cy": y, "r": 1.6,
                      "stroke": color, "width": 0.5, "fill": color})
    fp = resolved.get("footprint")
    if fp:
        items.append({"type": "polygon", "points": fp, "stroke": AMBER, "width": 0.8, "fill": "none"})
    label = resolved.get("label")
    if label and resolved.get("holes"):
        hx = min(h[0] for h in resolved["holes"])
        hy = min(h[1] for h in resolved["holes"])
        items.append({"type": "text", "x": hx, "y": hy - 5, "size": 5, "color": color, "text": label})
    return {"items": items}
