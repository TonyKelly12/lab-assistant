# Running it day to day

Two ways to make the on/off-daily routine painless: let **systemd start everything
on boot** (so "power on" is the whole startup), and a **web console** to glance at
status, restart anything stuck, and fire demo beats from a phone or wall tablet.

## One-time per node

Run once on each node (and again after a `git pull` to pick up unit changes). It
rewrites the unit's `User=`/`WorkingDirectory=` to this account and clone path and
enables everything to start on boot:

```bash
# Orin (broker + router + OLED), with the voice service and the console:
sudo bash scripts/install-services.sh orin --voice --console

# Shop Pi (camera + projector + bench-vision), with the voice satellite + console:
sudo bash scripts/install-services.sh shop --voice --console

# Homelab (the brain bridge):
sudo bash scripts/install-services.sh homelab
```

Then drop secrets into the env stubs it created and restart:

```bash
sudoedit /etc/default/lab-router        # MQTT_PASS=...
sudoedit /etc/default/shopvision        # MQTT_PASS=...
sudoedit /etc/default/lab-dashboard     # MQTT_PASS=...  and  DASHBOARD_KEY=...
cp -n config/dashboard.example.yaml config/dashboard.yaml   # set node_name/services/peers
```

On the Orin, also make sure `mosquitto` and `ollama` start on boot (the installer
enables them if their units exist). Pull the router model once: `ollama pull qwen2.5:3b`.

## Daily

Turn the Pi on. That's it — every service is a `systemctl enable`d unit, so boot
brings up the broker, router, cameras, projector render, and bench-vision in order.
To confirm at a glance, open the console (below) or:

```bash
systemctl is-active lab-router camera-node render-node shopvision
```

**Calibration survives reboots** — the projector mapping is a file on disk. Re-run it
only if the projector or camera physically moved:
`python3 calibration/projector_calibrate.py --config config/bench.yaml --bench-only`.
The projector display is a Chromium kiosk in the desktop session, not a systemd service
(systemd has no display to draw on). `install-services.sh shop` sets up an autostart
entry (`~/.config/autostart/lab-kiosk.desktop`) for it, so it comes up with the desktop —
this needs **Desktop Autologin** (`raspi-config`) so a GUI session actually runs on the
projector. To start it now without a reboot: `DISPLAY=:0 bash render_node/kiosk.sh`. If the
projector is blank, the page just has no scene yet; if the outline is offset from the real
board, re-run the calibration above. Wrong output/mode → edit `KIOSK_OUTPUT`/`KIOSK_MODE`
in `render_node/kiosk.sh` (`DISPLAY=:0 xrandr` lists outputs).

## The console

`http://<node>:8099` — one page showing every node's services as green/red LEDs, with
Start / Restart / Stop on each, a read-only health strip for the camera/projector, and
a **Demo deck** that fires the storyboard beats over MQTT (great from a tablet while you
film). Point the hub (e.g. the Orin) at the others with `peers:` in `config/dashboard.yaml`
and it drives all of them from the one page.

- **Status** is read-only and open on the LAN. **Control** (start/stop/restart and firing
  demo beats) needs the key — type your `DASHBOARD_KEY` into the Key box once; it's
  remembered in the browser.
- Control is least-privilege: the installer writes `/etc/sudoers.d/lab-assistant` allowing
  the console user to `systemctl start/stop/restart` **only** the lab units, nothing else.
- Bind it to your LAN only. It can start and stop services, so don't expose it publicly.

## If you'd rather not use systemd

Everything still runs by hand, one per SSH session, in this order: broker → router →
camera-node → render-node → shopvision (→ voice). See `README.md` and `docs/DEMO.md`.
