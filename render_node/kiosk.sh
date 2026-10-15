#!/usr/bin/env bash
# Opens the projector page fullscreen. Run from the Pi's desktop session
# (or add it to autostart; see README). The projector must be the active display.
set -euo pipefail
URL="${1:-http://localhost:8090/}"
OUTPUT="${KIOSK_OUTPUT:-HDMI-1}"      # the projector's output name in `xrandr`
MODE="${KIOSK_MODE:-1920x1080}"

# Make sure the projector output is switched on at the right mode. After a
# reboot the desktop sometimes leaves it connected but off (no picture).
if command -v xrandr >/dev/null && [ -n "${DISPLAY:-}" ]; then
  if ! xrandr --output "$OUTPUT" --mode "$MODE" --rate 60 2>/dev/null; then
    xrandr --output "$OUTPUT" --mode "$MODE" 2>/dev/null \
      || echo "kiosk: could not set $OUTPUT to $MODE; check: DISPLAY=:0 xrandr" >&2
  fi
fi
BROWSER="$(command -v chromium-browser || command -v chromium)"
exec "$BROWSER" \
  --kiosk --app="$URL" \
  --noerrdialogs --disable-infobars --no-first-run \
  --password-store=basic --disable-background-networking \
  --force-device-scale-factor=1 \
  --disable-session-crashed-bubble --disable-features=Translate \
  --check-for-update-interval=31536000
