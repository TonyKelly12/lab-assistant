# Demo runbook — "the bench that sees your board"

A single JARVIS-style sequence: voice in the shop → the bench finds your real
breadboard and lights up the exact holes → a hard question goes up to the
homelab brain and the answer comes back spoken and drawn. Editing is fine; this
runbook makes each beat repeatable.

## Who runs what

| Node | Service | Command |
|---|---|---|
| shop Pi | camera node | `python3 -m camera_node.service --config config/camera-node.yaml` |
| shop Pi | render node (projector) | `python3 -m render_node.server --config config/render-node.yaml` + `./render_node/kiosk.sh` |
| shop Pi | **bench-vision** | `python3 -m shopvision.service --config config/shopvision.yaml --demo` |
| shop Pi | **voice satellite** | `python3 shop/voice_satellite.py` |
| Orin | broker (Mosquitto) + router | `python3 orin/router.py --config config/orin.yaml` |
| Orin | **voice service** | `python3 orin/voice_service.py --warm` |
| homelab | **brain bridge** | `python3 homelab/brain_bridge.py` |

## One-time setup (per node)

- **shop Pi (bench-vision)** — already has OpenCV/NumPy/Flask from the camera node.
  `cp config/shopvision.example.yaml config/shopvision.yaml` and point it at the
  camera/render URLs and the saved `calibration/bench.json`.
- **shop Pi (satellite)** — `pip3 install sounddevice openwakeword numpy flask` and
  `sudo apt install alsa-utils` (for `aplay`). Env → `/etc/default/lab-voice-satellite`
  (see `config/voice.example.env`).
- **Orin (voice)** — `pip3 install faster-whisper piper-tts paho-mqtt flask`; download a
  Piper voice (e.g. `en_US-amy-medium`). Env → `/etc/default/lab-voice`.
- **homelab (bridge)** — `pip3 install paho-mqtt`; mint an agent token (control plane →
  Tokens, scope `api:read`). Env → `/etc/default/lab-brain-bridge`.

## Pre-flight (do this before rolling)

1. **Projector focus + calibration.** Focus chart, then
   `python3 calibration/projector_calibrate.py --config config/bench.yaml`; confirm the
   yellow crosses land on the markers.
2. **Projection works:** `python3 tools/send_scene.py --render-url http://<render-pi>:8090 holes 100 100 30 10`.
3. **Board finder lands:** `curl "http://<shop-pi>:8070/board?draw=1"` → the cyan outline
   sits on the breadboard; note the reported size (confirms full vs half profile).
4. **Each spoken line transcribes** — dry-run the router with no hardware:
   `python3 orin/router.py --config config/orin.yaml try "light up column 12"`.
5. **Tune the hole fit once** if dots sit high/low: adjust `rows`/`col_start` in the
   chosen profile in `labassist/breadboard.py` and re-check `/board?ref=row a&draw=1`.

## Dry-run each track (no voice)

```bash
# A — board-aware projection (publish straight to bench-vision):
mosquitto_pub -h <orin> -u lab -P "$MQTT_PASS" -t lab/board/request -m '{"ref":"column 12","id":"t1"}'
mosquitto_pub -h <orin> -u lab -P "$MQTT_PASS" -t lab/board/request -m '{"ref":"highlight the power rails","id":"t2"}'
mosquitto_pub -h <orin> -u lab -P "$MQTT_PASS" -t lab/board/request -m '{"ref":"place the ESP32 at column 10","id":"t3"}'

# C — brain finale (round-trips through the homelab):
mosquitto_pub -h <orin> -u lab -P "$MQTT_PASS" -t lab/brain/request -m '{"id":"b1","text":"why wont my ESP32 boot from USB?"}'

# B — voice end to end (satellite running): just say "jarvis, light up column 12".
#     Filming fallback: python3 shop/voice_satellite.py --ptt   (press Enter, then speak)
```

## Storyboard (shot list + exact lines)

1. **Cold open.** Bench dark, OLED cycling stats. *"Jarvis, turn on the bench lights."*
   → lights + exhaust react (existing rules). Establishes voice is live.
2. **The reveal.** Drop a breadboard at an angle. *"Jarvis, find my board."*
   → outline + size snap onto it.
3. **The hero.** *"Light up column 12."* → *"Highlight the power rails."* →
   *"Place the ESP32 at column 10."* Glowing marks land on the real holes; spoken
   confirmations from bench-vision.
4. **The brain.** *"Why won't my ESP32 boot from USB?"* → routes to the homelab brain
   → spoken answer + a short headline drawn on the bench.
5. **Command center.** Cut to the control-plane screen: agents live (incl. the Orin
   router), models running. *"One bench. Two brains."*

## Robustness notes
- Run bench-vision with `--demo`: if a live find misses on a take, it reuses the last
  good board pose so the projection still lands and you can keep rolling.
- The router only speaks a reply once; board + brain replies are spoken by their own
  services (same request id), so there's no double-talk.
- If voice STT fumbles a word on camera, retake with `--ptt` and dub in edit — the
  downstream behaviour is identical.
