# Circuit projection

Drop a breadboard on the bench and say **"wire up an ESP32 to an LED"** — the
bench finds your board and projects the whole layout onto the real holes: where
each part sits, and a colour-coded line for every wire. You follow the guides.

This is a **wiring plan projected onto the holes**, authored ahead of time. The
bench does *not* look at your parts or detect what you've built — it knows where
the board is (the markerless finder) and turns a layout into marks on the real
holes. Checking what you physically built against the plan is a separate, later
capability (the vision/"training" tier).

## Using it

Spoken / typed through the router (Orin):

| You say | What happens |
|---|---|
| "wire up an ESP32 to an LED" | finds the board, projects the full `esp32-led-blink` layout |
| "show me how to connect a voltage divider" | projects `voltage-divider` |
| "next" / "back" | step through the wires one at a time |
| "show the whole thing" | reveal every wire at once |
| "clear the board" | wipe it |

Tuning / B-roll without voice, straight at the bench-vision service:

```
curl "http://<shop-pi>:8070/circuits"                       # list the library
curl "http://<shop-pi>:8070/circuit?query=esp32+led&draw=1" # resolve + project
curl "http://<shop-pi>:8070/circuit?name=voltage-divider&step=1&draw=1"
```

## Two breadboards

- **Snapped together** (the dovetail locks, into one bigger working surface):
  the camera just sees a longer rectangle, so the holes are numbered
  **continuously** across the seam at the standard 2.54 mm pitch — it's treated
  as one board with more columns. Nothing to configure; it follows the measured
  size.
- **Sitting apart** (you run jumpers between them): a layout can place parts on
  each board with a per-hole `"board": 0|1` index, and wires that cross between
  them are drawn across the gap — you run the real jumper. See
  `circuits/esp32-sensor-2board.json`.

## Adding a circuit

Drop a JSON file in `circuits/`. Minimal shape:

```json
{
  "name": "my-circuit",
  "title": "Human name",
  "aliases": ["words", "people might say"],
  "summary": "one line",
  "board": "half",          // row geometry: "half" or "full" (omit = auto by size)
  "boards": 1,              // how many separate boards this layout expects
  "components": [
    {"ref": "U1", "part": "ESP32-S3", "span": [1, 16], "rows": ["a", "j"],
     "pins": [{"name": "GPIO2", "col": 8, "row": "j"}]},
    {"ref": "D1", "part": "LED", "pins": [
       {"name": "anode", "col": 24, "row": "i"},
       {"name": "cathode", "col": 26, "row": "i"}]}
  ],
  "wires": [
    {"from": {"col": 8, "row": "j"}, "to": {"col": 20, "row": "j"},
     "net": "signal", "label": "GPIO2 -> R1"}
  ]
}
```

- A **hole** is `{"col": <1-based>, "row": "a".."j"}`, or `{"col": N, "rail": "top+"}`
  for a power rail, plus an optional `"board": <index>` for the separate-boards case.
- `span` + `rows` draw a component's footprint outline; `pins` drop labelled dots.
- `net` picks the wire colour (power = red, gnd = grey, signal = amber, data =
  cyan); or set `"color"` explicitly.
- Wire order is the **step order** for "next".

Pin columns in the shipped circuits are a **template** — nudge them to match your
exact parts. The resolver range-checks every hole against the detected board, so
a bad reference is reported, never projected wrong. `python3 tests/test_circuit.py`
proves wire endpoints land on the referenced holes.
