# lab-assistant

Projection-mapped workbench assistant. This repo holds the parts that run in
the shop: camera nodes, projector render nodes and calibration. Nothing here
depends on the main brain; everything talks over HTTP, and over MQTT once a
broker is configured.

```
camera_node/     camera service: snapshots over HTTP, marker + motion tracking
render_node/     projector server + fullscreen page (draws in bench millimetres)
calibration/     printable markers, lens calibration, projector calibration
tools/           send_scene.py: draw things on the bench from the command line
labassist/       shared helpers (ArUco, homography maths, MQTT, config)
config/          *.example.yaml: copy to *.yaml and edit (your copies are git-ignored)
orin/            Orin: router.py (sentence -> action), oled_stats.py (case screen)
homelab/         tools for the homelab AI: lab_tools.py + an MCP server for Hermes
systemd/         service files for running at boot
tests/           offline tests with synthetic images (no hardware needed)
docs/            ARCHITECTURE.md (how it all works), CALIBRATION.md (calibration in detail)
```

## How it fits together

```
 Pi 4 #1 (camera node, :8080)          Pi 4 #2 (render node, :8090)
   bench-overhead  (ribbon cam)          Chromium kiosk -> Mooka projector
   bench-microscope (TOMLOV, USB)        draws scenes in bench mm, warped by
   GET /snapshot/<cam>.jpg               the calibration homography
   GET /markers/<cam>                    POST /scene  /pattern  /calibration
            \                                   /
             \---- calibration/projector_calibrate.py (any machine) ----/
```

Coordinates on the bench are millimetres: origin at printed marker 0 (back
left), x to the right, y toward you.

## 1. Install on both Pis

Raspberry Pi OS (64-bit). Install packages with apt, not pip (pip versions of
numpy/OpenCV break picamera2):

```bash
sudo apt update
sudo apt install -y git python3-opencv python3-numpy python3-flask python3-yaml python3-paho-mqtt v4l-utils
git clone git@github.com:TonyKelly12/lab-assistant.git ~/lab-assistant
cd ~/lab-assistant
python3 -c "import cv2; print(cv2.__version__, hasattr(cv2, 'aruco'))"   # must print True
```

Camera Pi also needs `sudo apt install -y python3-picamera2`.
Render Pi needs the desktop version of Raspberry Pi OS (for Chromium).

## 2. Camera node (Pi 4 #1)

```bash
cp config/camera-node.example.yaml config/camera-node.yaml
v4l2-ctl --list-devices          # find the TOMLOV's /dev/videoN, put it in the config
python3 -m camera_node.service --config config/camera-node.yaml
```

Check from any browser on your network:
- `http://<camera-pi>:8080/snapshot/bench-overhead.jpg`
- `http://<camera-pi>:8080/snapshot/bench-microscope.jpg`
- `http://<camera-pi>:8080/markers/bench-overhead`

If the microscope isn't plugged in, the service logs a warning and keeps the
overhead camera running.

## 3. Render node (Pi 4 #2, drives the Mooka)

Set the Pi's HDMI output to exactly 1920x1080 (Screen Configuration tool).

```bash
cp config/render-node.example.yaml config/render-node.yaml
python3 -m render_node.server --config config/render-node.yaml
./render_node/kiosk.sh            # in the desktop session, on the projector
```

Until calibrated, the page shows a "not calibrated" note and draws scenes in a
plain centred box. Test it from any machine:

```bash
python3 tools/send_scene.py --render-url http://<render-pi>:8090 rect 100 80 200 120
```

## 4. Print and place the markers

```bash
python3 calibration/generate_markers.py --out prints/
```

Print `prints/bench-markers.png` and `prints/checkerboard-9x6.png` at **100% /
actual size**, and check the 100 mm bar with a ruler.

No printer? Laser them instead: `prints/bench-markers.svg` and
`prints/checkerboard-9x6.svg` are exact-size vectors in millimetres (black fill =
engrave, red outline = cut). Engrave dark on a light, matte material such as
plywood, cardstock or white acrylic. Keep the white margin around each marker. Cut out the four bench
markers and tape them flat at the corners of the area you want to project on
(ID 0 back-left, 1 back-right, 2 front-right, 3 front-left), inside the
projected image and in view of the camera. Measure each marker's centre from
marker 0's centre and enter them in `config/bench.yaml`
(copy from `config/bench.example.yaml`).

## 5. Lens calibration (once per camera, optional but recommended)

Glue the checkerboard to something flat (cardboard, acrylic). Then:

```bash
python3 calibration/lens_calibrate.py --camera-url http://<camera-pi>:8080 --camera bench-overhead
```

Press Enter for each pose: centre, each corner, some tilted. Aim for RMS
under 1 px. Saves `calibration/intrinsics-bench-overhead.json`, which
`config/bench.yaml` points to.

## 6. Projector calibration

First focus the projector. Put up the focus chart, turn the Mooka's focus ring
until the star in the middle (and the 1-px line patches beside it) are crisp
where you work, check the corners, then clear it:

```bash
python3 tools/send_scene.py --render-url http://<render-pi>:8090 focus
python3 tools/send_scene.py --render-url http://<render-pi>:8090 clear
```

Don't touch the focus ring after calibrating: refocusing shifts and resizes the
image slightly, so it means a full re-calibration. A strip of tape over the
ring helps.

Then dim the room light, clear the bench as much as you can, and run:

```bash
python3 calibration/projector_calibrate.py --config config/bench.yaml
```

It lights the bench white to find the printed markers, projects a marker grid,
works out the mapping, and sends it to the render node. Yellow crosses should
land on the printed markers. If something fails, add `--save-images` and look
at `calibration/capture-*.jpg`.

**Tripod bumped?** If the camera is attached to the projector (so they moved
together), re-run with `--bench-only`: it only re-finds the printed markers,
which takes a few seconds.

**Milestone check:** project a breadboard hole grid onto a real breadboard:

```bash
python3 tools/send_scene.py --render-url http://<render-pi>:8090 holes <x_mm> <y_mm> 30 10
```

## Run at boot

Edit `User` and `WorkingDirectory` in `systemd/*.service` if needed, then:

```bash
sudo cp systemd/camera-node.service /etc/systemd/system/     # camera Pi
sudo cp systemd/render-node.service /etc/systemd/system/     # render Pi
sudo systemctl daemon-reload
sudo systemctl enable --now camera-node    # or render-node
```

For the kiosk, add `~/lab-assistant/render_node/kiosk.sh` to the desktop
session's autostart.

## Orin intent router

`orin/router.py` turns a sentence into an action: simple rules first, then the
local model (Ollama, fixed JSON reply), and anything needing real thought goes
to the main brain on `lab/brain/request`. Every decision is logged to
`logs/router.jsonl` for fine-tuning later.

```bash
cp config/orin.example.yaml config/orin.yaml        # edit the password and devices
python3 orin/router.py try "turn on the exhaust fan"     # see the decision, sends nothing
sudo cp systemd/lab-router.service /etc/systemd/system/  # edit User/paths first
sudo systemctl daemon-reload && sudo systemctl enable --now lab-router
mosquitto_pub -h localhost -u lab -P '...' -t lab/router/command -m "show me a breadboard grid"
```

## Homelab tools (Hermes)

`homelab/lab_tools.py` has the lab as plain Python functions (draw, clear,
switch devices, camera snapshot, markers, ask the router); `homelab/mcp_server.py`
serves them over MCP. Settings come from `LAB_*` environment variables, and
`config/mosquitto-acl.example` gives the homelab its own limited broker login.

## Orin case screen

`orin/oled_stats.py` drives the Yahboom CUBE Nano case's OLED. It shows three
pages in turn: system (CPU, RAM, disk, IP), Jetson (GPU load, power mode,
temps, fan) and lab services (MQTT, Ollama, voice, last MQTT message). It
dims overnight, shifts the picture a pixel now and then against burn-in, and
flashes a warning when the Tj temperature reaches 85 C.

```bash
pip3 install smbus Adafruit_SSD1306 Pillow paho-mqtt
sudo pip3 install jetson-stats             # optional but better; reboot after
python3 orin/oled_stats.py --preview /tmp/page.png   # check the layout, no hardware
python3 orin/oled_stats.py --debug                   # run it; Ctrl+C to stop
sudo cp systemd/oled-stats.service /etc/systemd/system/   # edit User/paths first
sudo systemctl daemon-reload && sudo systemctl enable --now oled-stats
```

Put the MQTT login in `/etc/default/oled-stats` (`MQTT_USER=lab`,
`MQTT_PASS=...`) so the "last message" line works. `--help` lists the
options (night hours, alert temperature, page time, RGB effect).

## Board-aware projection, voice, and the brain bridge (demo)

Three services build the JARVIS demo on top of everything above. Full setup,
storyboard and dry-run commands are in `docs/DEMO.md`. For day-to-day startup
(auto-start on boot + a web console to run/restart services and fire demo beats
from a tablet), see `docs/OPERATING.md` and `scripts/install-services.sh`.

- **Bench-vision** (`shopvision/`, shop Pi) — the hero. Finds the real breadboard
  and projects onto the actual holes. Listens on MQTT `lab/board/request
  {"ref": "column 12"}`; `ref` can be a column, row, hole (`e12`), the power rails,
  "find my board", or "place the ESP32 at column N". `labassist/breadboard.py` holds
  the hole geometry (tune the profile once against `/board?ref=…&draw=1`). The Orin
  router forwards board-ish speech here automatically (`highlight_board` intent).
- **Voice** (`orin/voice_service.py` + `shop/voice_satellite.py`) — the satellite does
  wake word + capture and POSTs the clip to the Orin, which runs Whisper + Piper and
  speaks every reply back. `--ptt` on the satellite is a push-to-talk fallback.
- **Brain bridge** (`homelab/brain_bridge.py`, homelab) — answers `lab/brain/request`
  with the main-brain RAG and replies on `lab/brain/reply`; the router speaks it and
  draws a headline on the bench.

## Scene format

`POST /scene` (or MQTT `lab/display/bench`) with `{"items": [...], "append": false}`.
All positions and sizes in millimetres.

| type | fields |
|---|---|
| rect | x, y, w, h, r?, stroke?, fill?, width? |
| circle | cx, cy, r, stroke?, fill?, width? |
| line | x1, y1, x2, y2, stroke?, width? |
| polyline / polygon | points: [[x, y], ...] |
| text | x, y, text, size?, color?, anchor?, bold? |
| image | href, x, y, w, h |
| grid | x, y, w, h, pitch? (2.54), dots?, dot? |
| cross | x, y, size? |

## Tests

```bash
python3 tests/test_calibration.py   # synthetic camera + projector, checks accuracy
python3 tests/test_services.py      # camera and render servers with fake inputs
```
