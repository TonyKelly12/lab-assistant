#!/usr/bin/env bash
# Opens the projector page fullscreen. Run from the Pi's desktop session
# (or add it to autostart; see README). The projector must be the active display.
set -euo pipefail
URL="${1:-http://localhost:8090/}"
BROWSER="$(command -v chromium-browser || command -v chromium)"
exec "$BROWSER" \
  --kiosk --app="$URL" \
  --noerrdialogs --disable-infobars --no-first-run \
  --password-store=basic --disable-background-networking \
  --force-device-scale-factor=1 \
  --disable-session-crashed-bubble --disable-features=Translate \
  --check-for-update-interval=31536000
