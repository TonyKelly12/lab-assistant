#!/usr/bin/env python3
"""Voice satellite.  NODE: shop Pi (mic + speaker at the bench).

The ears and mouth in the shop. It listens locally for the wake word, records
the utterance, and sends just that clip to the Orin voice service (so constant
audio never leaves the shop). The Orin sends spoken replies back to the /play
endpoint here, which plays them on the bench speaker.

    wake word ("jarvis") + silence-endpointed record
        -> POST WAV to  {ORIN_URL}/utterance
    POST /play  (WAV)  -> play on the shop speaker (aplay)

A push-to-talk fallback (press Enter) is handy while filming. Audio libs load
lazily so the module imports without them for a syntax check.

Env:
    ORIN_URL        e.g. http://orin.local:8095   (the voice service)
    WAKE_MODEL      openWakeWord model name/path (default "hey jarvis")
    WAKE_THRESHOLD  default 0.5
    SAT_HTTP_PORT   default 8096
    MIC_RATE        default 16000      MIC_DEVICE / SPK_DEVICE  optional indices
    SILENCE_RMS     default 500        SILENCE_HOLD_S  default 0.8   MAX_UTTER_S 10
    PLAY_CMD        default "aplay -q {path}"

Run:  python3 shop/voice_satellite.py            # wake-word mode
      python3 shop/voice_satellite.py --ptt      # push-to-talk (Enter to talk)
"""
import argparse
import logging
import os
import subprocess
import tempfile
import threading
import urllib.request
import wave

from flask import Flask, jsonify, request

log = logging.getLogger("satellite")

ORIN_URL = os.environ.get("ORIN_URL", "http://orin.local:8095").rstrip("/")
RATE = int(os.environ.get("MIC_RATE", "16000"))
CHUNK = 1280                      # 80 ms at 16 kHz (openWakeWord frame size)
SILENCE_RMS = int(os.environ.get("SILENCE_RMS", "500"))
SILENCE_HOLD_S = float(os.environ.get("SILENCE_HOLD_S", "0.8"))
MAX_UTTER_S = float(os.environ.get("MAX_UTTER_S", "10"))
PLAY_CMD = os.environ.get("PLAY_CMD", "aplay -q {path}")


def _rms(frame):
    import numpy as np
    a = np.frombuffer(frame, dtype="int16").astype("float32")
    return float((a ** 2).mean() ** 0.5) if a.size else 0.0


def _wav_bytes(frames):
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
    return buf.getvalue()


def send_utterance(frames):
    wav = _wav_bytes(frames)
    req = urllib.request.Request(f"{ORIN_URL}/utterance", data=wav,
                                 headers={"Content-Type": "audio/wav"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            log.info("sent %d ms -> %s", int(1000 * len(frames) * CHUNK / RATE), r.read()[:200])
    except Exception as exc:  # noqa: BLE001
        log.warning("utterance post failed: %s", exc)


def record_utterance(stream):
    """Record from an open input stream until ~SILENCE_HOLD_S of quiet."""
    frames, silent, spoke = [], 0.0, False
    per_chunk = CHUNK / RATE
    while len(frames) * per_chunk < MAX_UTTER_S:
        frame = stream.read(CHUNK)[0].tobytes()
        frames.append(frame)
        if _rms(frame) > SILENCE_RMS:
            spoke, silent = True, 0.0
        else:
            silent += per_chunk
        if spoke and silent >= SILENCE_HOLD_S:
            break
    return frames


def listen_loop(ptt=False):
    import sounddevice as sd
    device = os.environ.get("MIC_DEVICE")
    device = int(device) if device else None
    with sd.RawInputStream(samplerate=RATE, blocksize=CHUNK, dtype="int16",
                           channels=1, device=device) as stream:
        if ptt:
            log.info("push-to-talk: press Enter to talk")
            while True:
                input()
                send_utterance(record_utterance(stream))
            return
        from openwakeword.model import Model
        oww = Model(wakeword_models=[os.environ.get("WAKE_MODEL", "hey jarvis")])
        threshold = float(os.environ.get("WAKE_THRESHOLD", "0.5"))
        log.info("listening for the wake word")
        import numpy as np
        while True:
            frame = stream.read(CHUNK)[0].tobytes()
            scores = oww.predict(np.frombuffer(frame, dtype="int16"))
            if any(v >= threshold for v in scores.values()):
                log.info("wake word")
                oww.reset()
                send_utterance(record_utterance(stream))


def play(wav_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        path = f.name
    try:
        subprocess.run(PLAY_CMD.format(path=path).split(), check=False)
    finally:
        os.unlink(path)


def create_app():
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify(ok=True, orin=ORIN_URL)

    @app.post("/play")
    def play_endpoint():
        data = request.get_data()
        if not data:
            return jsonify(ok=False, error="empty"), 400
        threading.Thread(target=play, args=(data,), daemon=True).start()
        return jsonify(ok=True, bytes=len(data))

    return app


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ptt", action="store_true", help="push-to-talk instead of wake word")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    threading.Thread(target=listen_loop, kwargs={"ptt": a.ptt}, daemon=True).start()
    port = int(os.environ.get("SAT_HTTP_PORT", "8096"))
    log.info("satellite playback on :%d, talking to %s", port, ORIN_URL)
    create_app().run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    main()
