"""Offline proof that circuit layouts land on the real holes.

No hardware, no camera: we feed a synthetic detected board (a known rectangle in
bench mm) and assert that every wire endpoint the renderer emits is exactly the
coordinate `breadboard.hole_xy` gives for the hole the layout referenced. If
these pass, the projector draws the plan on the real holes once calibration maps
bench mm to projector pixels.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from labassist import breadboard as bb  # noqa: E402
from labassist import circuit as ckt  # noqa: E402

CIRCUITS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "circuits")


def half_board(origin=(0.0, 0.0)):
    """A 55 x 83 mm half breadboard, axis-aligned, long side along +x."""
    ox, oy = origin
    return {
        "center": [ox + 41.5, oy + 27.5],
        "size": [55.0, 83.0],
        "angle": 0.0,
        "corners": [[ox, oy], [ox + 83.0, oy], [ox + 83.0, oy + 55.0], [ox, oy + 55.0]],
        "fill": 0.9,
    }


def _lines(scene):
    return [it for it in scene["items"] if it["type"] == "line"]


def test_resolve_hole_matches_hole_xy():
    board = half_board()
    prof = bb.pick_profile(board, name="half")
    want = bb.hole_xy(board, prof, 10, prof.rows["e"])
    got = ckt.resolve_hole([board], [prof], {"col": 10, "row": "e"})
    assert got == want


def test_blink_wire_endpoints_sit_on_referenced_holes():
    board = half_board()
    prof = bb.pick_profile(board, name="half")
    layout = ckt.load_circuit("esp32-led-blink", CIRCUITS)
    assert ckt.validate(layout, [board], [prof]) == []
    scene = ckt.render_circuit(layout, [board], [prof])
    lines = _lines(scene)
    assert len(lines) == len(layout["wires"]) == scene["steps"]
    for wire, line in zip(layout["wires"], lines):
        a = ckt.resolve_hole([board], [prof], wire["from"])
        b = ckt.resolve_hole([board], [prof], wire["to"])
        assert [line["x1"], line["y1"]] == a
        assert [line["x2"], line["y2"]] == b


def test_step_through_reveals_wires_incrementally():
    board = half_board()
    prof = bb.pick_profile(board, name="half")
    layout = ckt.load_circuit("esp32-led-blink", CIRCUITS)
    assert len(_lines(ckt.render_circuit(layout, [board], [prof], steps=0))) == 0
    assert len(_lines(ckt.render_circuit(layout, [board], [prof], steps=1))) == 1
    full = ckt.render_circuit(layout, [board], [prof], steps=99)
    assert len(_lines(full)) == len(layout["wires"])
    assert full["shown"] == full["steps"]


def test_validate_flags_bad_column_and_board():
    board = half_board()
    prof = bb.pick_profile(board, name="half")
    bad = {"boards": 1, "wires": [
        {"from": {"col": 99, "row": "a"}, "to": {"col": 1, "row": "a"}},   # col out of range
        {"from": {"col": 1, "row": "a"}, "to": {"col": 1, "row": "z"}},    # bad row
        {"from": {"board": 5, "col": 1, "row": "a"}, "to": {"col": 1, "row": "a"}},  # no such board
    ]}
    problems = ckt.validate(bad, [board], [prof])
    assert len(problems) >= 3


def test_two_boards_resolve_in_shared_frame():
    b0 = half_board(origin=(0.0, 0.0))
    b1 = half_board(origin=(0.0, 70.0))   # second board 70 mm north, same frame
    layout = ckt.load_circuit("esp32-sensor-2board", CIRCUITS)
    profs = ckt.profiles_for([b0, b1], layout)
    assert ckt.validate(layout, [b0, b1], profs) == []
    scene = ckt.render_circuit(layout, [b0, b1], profs)
    lines = _lines(scene)
    # every cross-board wire should have endpoints on each board's own holes
    for wire, line in zip(layout["wires"], lines):
        a = ckt.resolve_hole([b0, b1], profs, wire["from"])
        b = ckt.resolve_hole([b0, b1], profs, wire["to"])
        assert [line["x1"], line["y1"]] == a
        assert [line["x2"], line["y2"]] == b
    # board 1 holes really are offset onto the second board (y >= 70)
    s1_vcc = ckt.resolve_hole([b0, b1], profs, {"board": 1, "col": 3, "row": "j"})
    assert s1_vcc[1] >= 70.0


def test_two_board_layout_needs_two_boards():
    board = half_board()
    layout = ckt.load_circuit("esp32-sensor-2board", CIRCUITS)
    problems = ckt.validate(layout, [board])
    assert any("expects 2" in p for p in problems)


def joined_surface():
    """Two half boards snapped end-to-end: one 55 x 166 mm continuous surface."""
    return {
        "center": [83.0, 27.5],
        "size": [55.0, 166.0],
        "angle": 0.0,
        "corners": [[0.0, 0.0], [166.0, 0.0], [166.0, 55.0], [0.0, 55.0]],
        "fill": 0.9,
    }


def test_snapped_surface_numbers_columns_continuously():
    board = joined_surface()
    # a single half board caps at 30 columns; the snapped surface must exceed it
    prof = ckt.bench_profile(board)
    ncols = bb._col_count(board, prof)
    assert ncols > 40, f"expected a continuous grid across the seam, got {ncols} columns"
    # a column that only exists on the far (second) board still resolves on-surface
    far = ckt.resolve_hole([board], [prof], {"col": 45, "row": "c"})
    assert far[0] > 83.0  # past the seam, on the second board's half
    bad = {"boards": 1, "wires": [{"from": {"col": 45, "row": "c"}, "to": {"col": 50, "row": "c"}}]}
    assert ckt.validate(bad, [board], [prof]) == []


def test_find_circuit_matches_natural_request():
    assert ckt.find_circuit("show me how to wire an esp32 to an led", CIRCUITS)["name"] == "esp32-led-blink"
    assert ckt.find_circuit("build a voltage divider", CIRCUITS)["name"] == "voltage-divider"
    assert ckt.find_circuit("make me a sandwich", CIRCUITS) is None
