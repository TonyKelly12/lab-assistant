#!/usr/bin/env python3
"""Voice service.  NODE: Orin (GPU).

Speech brain for the lab. The shop satellite captures an utterance and POSTs the
WAV here; this service transcribes it (faster-whisper on the GPU), feeds it to
the router as if spoken, and speaks every reply back by synthesising audio
(Piper) and pushing it to the satellite's speaker.

    POST /utterance   (WAV body or multipart 'file')  -> {"id","text"}  (transcript ack)
       -> publish  lab/router/command {text, source:"voice", id}
    subscribe lab/router/reply  -> if it answers one of our utterances, Piper TTS
       -> POST the WAV to the satellite's /play

Only replies to utterances this service started are spoken (so Hermes' own
`ask_lab` traffic on the bus stays silent). Heavy models load lazily, so the
module imports fine without them for tests.

Env:
    VOICE_WHISPER_MODEL   default "base.en"     VOICE_WHISPER_DEVICE  default "cuda"
    VOICE_WHISPER_COMPUTE default "float16"
    VOICE_PIPER_MODEL     path to a Piper .onnx voice (required for speech out)
    VOICE_PIPER_CONFIG    optional .onnx.json (defaults next to the model)
    SATELLITE_URL         e.g. http://shop-pi.local:8096  (where /play lives)
    LAB_MQTT_HOST/PORT/USER/PASS   the Orin broker (localhost by default)
    VOICE_HTTP_PORT       default 8095     VOICE_REPLY_TTL  default 40 (s)

Run:  python3 orin/voice_service.py
"""
import argparse
import io
import logging
import os
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
import wave

from flask import Flask, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labassist.mqtt import Bus  # noqa: E402

log = logging.getLogger("voice")

SATELLITE_URL = os.environ.get("SATELLITE_URL", "").rstrip("/")
REPLY_TTL = float(os.environ.get("VOICE_REPLY_TTL", "40"))

_whisper = None
_piper = None
_voice_ids = {}        # id -> time we issued it (so we only speak our own replies)
_lock = threading.Lock()


def whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel(
            os.environ.get("VOICE_WHISPER_MODEL", "base.en"),
            device=os.environ.get("VOICE_WHISPER_DEVICE", "cuda"),
            compute_type=os.environ.get("VOICE_WHISPER_COMPUTE", "float16"),
        )
        log.info("whisper model loaded")
    return _whisper


def piper():
    global _piper
    if _piper is None:
        from piper.voice import PiperVoice
        model = os.environ["VOICE_PIPER_MODEL"]
        config = os.environ.get("VOICE_PIPER_CONFIG") or (model + ".json")
        _piper = PiperVoice.load(model, config_path=config)
        log.info("piper voice loaded")
    return _piper


def transcribe(wav_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        path = f.name
    try:
        segments, _ = whisper().transcribe(
            path, beam_size=1, language="en",
            vad_filter=True,                     # strip silence/noise -> far fewer hallucinated words
            condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()
    finally:
        os.unlink(path)


def synthesize(text):
    """Return WAV bytes for `text` using Piper, or None if TTS isn't available
    (piper-tts not installed, or no voice model configured)."""
    try:
        voice = piper()
    except Exception as exc:  # noqa: BLE001
        log.warning("TTS unavailable (%s); skipping spoken reply", exc)
        return None
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voice.synthesize(text, wf)
    return buf.getvalue()


def push_to_satellite(wav_bytes):
    if not SATELLITE_URL:
        log.warning("SATELLITE_URL unset; dropping %d bytes of speech", len(wav_bytes))
        return
    req = urllib.request.Request(f"{SATELLITE_URL}/play", data=wav_bytes,
                                 headers={"Content-Type": "audio/wav"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=30).read()
    except Exception as exc:  # noqa: BLE001
        log.warning("playback push failed: %s", exc)


def _remember(request_id):
    now = time.time()
    with _lock:
        _voice_ids[request_id] = now
        for k, t in list(_voice_ids.items()):
            if now - t > REPLY_TTL:
                _voice_ids.pop(k, None)


def _is_ours(request_id):
    with _lock:
        t = _voice_ids.get(request_id)
    return t is not None and (time.time() - t) <= REPLY_TTL


def create_app(bus):
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify(ok=True, satellite=bool(SATELLITE_URL))

    @app.post("/utterance")
    def utterance():
        data = request.files["file"].read() if "file" in request.files else request.get_data()
        if not data:
            return jsonify(ok=False, error="empty audio"), 400
        try:
            text = transcribe(data)
        except Exception as exc:  # noqa: BLE001
            log.exception("transcribe failed")
            return jsonify(ok=False, error=str(exc)), 503
        rid = uuid.uuid4().hex[:12]
        if text:
            _remember(rid)
            bus.publish("lab/router/command", {"text": text, "source": "voice", "id": rid})
        log.info("utterance %s -> %r", rid, text)
        return jsonify(ok=True, id=rid, text=text)

    return app


def on_reply(topic, payload):
    if not isinstance(payload, dict):
        return
    rid, text = payload.get("id"), str(payload.get("text", "")).strip()
    if not text or not _is_ours(rid):
        return
    try:
        wav = synthesize(text)
        if wav:
            push_to_satellite(wav)
    except Exception:  # noqa: BLE001
        log.exception("speak failed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--warm", action="store_true", help="load the models at startup instead of on first use")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    bus = Bus({
        "enabled": True,
        "host": os.environ.get("LAB_MQTT_HOST", "127.0.0.1"),
        "port": int(os.environ.get("LAB_MQTT_PORT", "1883")),
        "username": os.environ.get("LAB_MQTT_USER"),
        "password": os.environ.get("LAB_MQTT_PASS"),
    }, client_id="orin-voice")
    if bus.client is None:
        sys.exit("MQTT is required for the voice service (install paho-mqtt and set LAB_MQTT_*)")
    bus.subscribe("lab/router/reply", on_reply)

    if a.warm:
        whisper()
        if os.environ.get("VOICE_PIPER_MODEL"):
            piper()

    port = int(os.environ.get("VOICE_HTTP_PORT", "8095"))
    log.info("voice service on :%d (satellite=%s)", port, SATELLITE_URL or "unset")
    create_app(bus).run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    main()
