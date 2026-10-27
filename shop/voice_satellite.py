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
    WAKE_MODEL      openWakeWord key or .onnx path (default "hey_jarvis";
                    spaces are normalised, so "hey jarvis" also works)
    WAKE_FRAMEWORK  "onnx" (default) or "tflite"
    WAKE_THRESHOLD  default 0.5
    SAT_HTTP_PORT   default 8096
    CAPTURE_RATE    mic's native rate, default 48000 (a Blue Yeti is fixed 48 kHz)
    CAPTURE_CHANNELS default 1 (set 2 if the mic refuses mono, e.g. some Yetis)
    MIC_DEVICE / SPK_DEVICE   optional device indices (see --list-devices)
    SILENCE_RMS     default 500        SILENCE_HOLD_S  default 0.8   MAX_UTTER_S 10
    PLAY_CMD        default "aplay -q {path}"

Run:  python3 shop/voice_satellite.py                 # wake-word mode
      python3 shop/voice_satellite.py --ptt           # push-to-talk (Enter to talk)
      python3 shop/voice_satellite.py --list-devices  # find the Yeti's index
      python3 shop/voice_satellite.py --meter          # live mic level + wake score
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
# Pipeline rate is fixed: the wake word (openWakeWord) and Whisper both require
# 16 kHz mono. The mic may capture at a different rate - e.g. a Blue Yeti is a
# fixed 48 kHz device and cannot hand us 16 kHz - so we capture at CAPTURE_RATE
# and resample to 16 kHz mono in software before anything downstream sees it.
RATE = 16000
CAPTURE_RATE = int(os.environ.get("CAPTURE_RATE", os.environ.get("MIC_RATE", "48000")))
CAPTURE_CHANNELS = int(os.environ.get("CAPTURE_CHANNELS", "1"))
CHUNK = 1280                      # 80 ms at 16 kHz (openWakeWord frame size)
SILENCE_RMS = int(os.environ.get("SILENCE_RMS", "500"))
SILENCE_HOLD_S = float(os.environ.get("SILENCE_HOLD_S", "0.8"))
MAX_UTTER_S = float(os.environ.get("MAX_UTTER_S", "10"))
PLAY_CMD = os.environ.get("PLAY_CMD", "aplay -q {path}")


def _rms(frame):
    import numpy as np
    a = frame.astype("float32") if hasattr(frame, "astype") else np.frombuffer(frame, dtype="int16").astype("float32")
    return float((a ** 2).mean() ** 0.5) if a.size else 0.0


def _wav_bytes(frames):
    """frames: a list of 16 kHz mono int16 ndarrays (or raw bytes)."""
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(f.tobytes() if hasattr(f, "tobytes") else f for f in frames))
    return buf.getvalue()


class Mic:
    """Opens the capture device at its native rate and yields a clean stream of
    16 kHz mono int16 frames of CHUNK samples, resampling/downmixing as needed.

    Capturing at the device's real rate (CAPTURE_RATE, 48 kHz for a Yeti) and
    resampling here means the mic's fixed rate never has to match the pipeline.
    """

    def __init__(self, capture_rate=CAPTURE_RATE, channels=CAPTURE_CHANNELS, device=None):
        import numpy as np
        self.np = np
        self.capture_rate = capture_rate
        self.channels = channels
        self.device = device
        self.read_block = max(CHUNK, int(round(CHUNK * capture_rate / RATE)))
        self._buf = np.zeros(0, dtype=np.float32)

    def _to_16k_mono(self, block):
        np = self.np
        a = block.astype(np.float32)
        if a.ndim == 2 and a.shape[1] > 1:   # downmix stereo -> mono
            a = a.mean(axis=1)
        else:
            a = a.reshape(-1)
        if self.capture_rate == RATE:
            return a
        n = a.shape[0]
        m = int(round(n * RATE / self.capture_rate))
        if m <= 0:
            return np.zeros(0, dtype=np.float32)
        return np.interp(np.linspace(0, n - 1, m), np.arange(n), a)

    def frames16k(self):
        import sounddevice as sd
        np = self.np
        with sd.InputStream(samplerate=self.capture_rate, channels=self.channels,
                            dtype="int16", blocksize=self.read_block, device=self.device) as stream:
            while True:
                block, _ = stream.read(self.read_block)
                self._buf = np.concatenate([self._buf, self._to_16k_mono(block)])
                while self._buf.shape[0] >= CHUNK:
                    out, self._buf = self._buf[:CHUNK], self._buf[CHUNK:]
                    yield np.clip(out, -32768, 32767).astype(np.int16)


def send_utterance(frames):
    wav = _wav_bytes(frames)
    samples = sum(len(f) for f in frames)
    req = urllib.request.Request(f"{ORIN_URL}/utterance", data=wav,
                                 headers={"Content-Type": "audio/wav"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            log.info("sent %d ms -> %s", int(1000 * samples / RATE), r.read()[:200])
    except Exception as exc:  # noqa: BLE001
        log.warning("utterance post failed: %s", exc)


def record_utterance(frames):
    """Pull 16 kHz mono frames until ~SILENCE_HOLD_S of quiet (or MAX_UTTER_S)."""
    out, silent, spoke = [], 0.0, False
    per_chunk = CHUNK / RATE
    for frame in frames:
        out.append(frame)
        if _rms(frame) > SILENCE_RMS:
            spoke, silent = True, 0.0
        else:
            silent += per_chunk
        if (spoke and silent >= SILENCE_HOLD_S) or len(out) * per_chunk >= MAX_UTTER_S:
            break
    return out


def load_wakeword():
    """Load openWakeWord, downloading the pretrained models on first run.

    openWakeWord's bundled keys use underscores ("hey_jarvis"), so a value like
    "hey jarvis" is normalised. Returns (model, name). On a fresh box the feature
    + pretrained models aren't present yet, so we fetch them and retry once.
    """
    import openwakeword
    from openwakeword.model import Model
    name = os.environ.get("WAKE_MODEL", "hey_jarvis").strip().replace(" ", "_")
    framework = os.environ.get("WAKE_FRAMEWORK", "onnx")
    try:
        return Model(wakeword_models=[name], inference_framework=framework), name
    except Exception as exc:  # noqa: BLE001
        log.warning("wake model %r not ready (%s); downloading pretrained models...", name, exc)
        try:
            openwakeword.utils.download_models()
        except Exception as dexc:  # noqa: BLE001
            log.error("could not download wake models: %s", dexc)
        return Model(wakeword_models=[name], inference_framework=framework), name


def meter_loop():
    """Print live mic level + wake-word score. No speaker needed: watch the
    numbers move as you talk and see how close 'hey jarvis' gets to threshold."""
    device = os.environ.get("MIC_DEVICE")
    device = int(device) if device else None
    frames = Mic(CAPTURE_RATE, CAPTURE_CHANNELS, device).frames16k()
    try:
        oww, name = load_wakeword()
    except Exception as exc:  # noqa: BLE001
        log.error("wake model failed to load: %s", exc)
        oww, name = None, "-"
    threshold = float(os.environ.get("WAKE_THRESHOLD", "0.5"))
    print(f"capture {CAPTURE_RATE} Hz {CAPTURE_CHANNELS}ch -> 16k mono | "
          f"wake '{name}' threshold {threshold}\ntalk now (Ctrl-C to stop)")
    for frame in frames:
        level = _rms(frame)
        score = max(oww.predict(frame).values()) if oww else 0.0
        bar = "#" * min(40, int(level / 200))
        hit = "  <-- WAKE" if score >= threshold else ""
        print(f"\rlevel {level:6.0f} |{bar:<40}| wake {score:4.2f}{hit}   ", end="", flush=True)


def listen_loop(ptt=False):
    device = os.environ.get("MIC_DEVICE")
    device = int(device) if device else None
    mic = Mic(CAPTURE_RATE, CAPTURE_CHANNELS, device)
    frames = mic.frames16k()      # one stateful stream; the recorder pulls from it too
    if ptt:
        log.info("push-to-talk: press Enter to talk (capture %d Hz %dch -> 16k mono)",
                 CAPTURE_RATE, CAPTURE_CHANNELS)
        while True:
            input()
            send_utterance(record_utterance(frames))
        return
    oww, model_name = load_wakeword()
    threshold = float(os.environ.get("WAKE_THRESHOLD", "0.5"))
    log.info("listening for '%s' (threshold %.2f, capture %d Hz %dch -> 16k mono)",
             model_name, threshold, CAPTURE_RATE, CAPTURE_CHANNELS)
    for frame in frames:
        scores = oww.predict(frame)
        if any(v >= threshold for v in scores.values()):
            log.info("wake word")
            oww.reset()
            send_utterance(record_utterance(frames))


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


def list_devices():
    import sounddevice as sd
    print(sd.query_devices())
    print("\nSet MIC_DEVICE / SPK_DEVICE to the index of the input/output you want.")
    print(f"Capture is {CAPTURE_RATE} Hz, {CAPTURE_CHANNELS} ch -> resampled to 16 kHz mono.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ptt", action="store_true", help="push-to-talk instead of wake word")
    p.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    p.add_argument("--meter", action="store_true", help="show live mic level + wake score, then exit")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if a.list_devices:
        list_devices()
        return
    if a.meter:
        try:
            meter_loop()
        except KeyboardInterrupt:
            print()
        return

    threading.Thread(target=listen_loop, kwargs={"ptt": a.ptt}, daemon=True).start()
    port = int(os.environ.get("SAT_HTTP_PORT", "8096"))
    log.info("satellite playback on :%d, talking to %s", port, ORIN_URL)
    create_app().run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    main()
