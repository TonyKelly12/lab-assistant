"""Camera sources: Pi ribbon camera (picamera2) and USB cameras (V4L2).

Each source gives two things:
  frame_small() -> grayscale uint8 image for continuous tracking (cheap)
  snapshot()    -> full-resolution BGR image for the AI / calibration
"""
import logging
import threading
import time

import cv2
import numpy as np

log = logging.getLogger("sources")


class PiCamSource:
    """Ribbon-cable camera through picamera2, with a main and a low-res stream."""

    def __init__(self, cfg):
        from picamera2 import Picamera2  # only needed on the Pi

        self.lock = threading.Lock()
        self.cam = Picamera2(cfg.get("index", 0))
        sensor_w, sensor_h = self.cam.sensor_resolution
        main = tuple(cfg.get("main_size") or (sensor_w, sensor_h))
        lores = tuple(cfg.get("lores_size", (640, 480)))
        config = self.cam.create_video_configuration(
            main={"size": main, "format": "RGB888"},  # RGB888 is BGR byte order, as OpenCV expects
            lores={"size": lores, "format": "YUV420"},
            buffer_count=int(cfg.get("buffer_count", 2)),
        )
        self.cam.configure(config)
        controls = cfg.get("controls") or {}
        if controls:
            self.cam.set_controls(controls)
        self.cam.start()
        time.sleep(1.0)  # let exposure settle
        self.main_size = main
        self.lores_size = lores
        log.info("picamera2 started: main %s, lores %s", main, lores)

    def frame_small(self):
        with self.lock:
            yuv = self.cam.capture_array("lores")
        return yuv[: self.lores_size[1], : self.lores_size[0]]  # Y plane = grayscale

    def snapshot(self):
        with self.lock:
            return self.cam.capture_array("main")

    def close(self):
        self.cam.stop()


class UsbSource:
    """USB (UVC) camera such as the TOMLOV microscope in PC camera mode.

    A background thread keeps grabbing so the newest frame is always ready;
    frames are only decoded when someone asks for one.
    """

    def __init__(self, cfg):
        self.device = cfg.get("device", "/dev/video0")
        self.size = tuple(cfg.get("size", (1920, 1080)))
        self.lock = threading.Lock()
        self.cap = None
        self._open()
        self.running = True
        threading.Thread(target=self._grab_loop, daemon=True).start()

    def _open(self):
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            log.warning("could not open %s; will retry", self.device)
        else:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            log.info("USB camera %s opened at %dx%d", self.device, w, h)
        self.cap = cap

    def _grab_loop(self):
        while self.running:
            with self.lock:
                ok = self.cap is not None and self.cap.isOpened() and self.cap.grab()
            if not ok:
                time.sleep(2)
                with self.lock:
                    if self.cap is not None:
                        self.cap.release()
                    self._open()
            else:
                time.sleep(0.01)

    def snapshot(self):
        with self.lock:
            ok, frame = self.cap.retrieve() if self.cap is not None else (False, None)
        if not ok or frame is None:
            raise RuntimeError(f"no frame from {self.device}")
        return frame

    def frame_small(self):
        frame = self.snapshot()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (640, int(640 * gray.shape[0] / gray.shape[1])))

    def close(self):
        self.running = False
        with self.lock:
            if self.cap is not None:
                self.cap.release()


class FakeSource:
    """Test source: serves an image file (or a synthetic pattern). For desk testing without a camera."""

    def __init__(self, cfg):
        path = cfg.get("image")
        if path:
            self.image = cv2.imread(path)
            if self.image is None:
                raise RuntimeError(f"could not read {path}")
        else:
            self.image = np.full((1944, 2592, 3), 40, np.uint8)
            cv2.putText(self.image, "fake camera", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 6, (200, 200, 200), 12)

    def snapshot(self):
        return self.image.copy()

    def frame_small(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (640, int(640 * gray.shape[0] / gray.shape[1])))

    def close(self):
        pass


def open_source(cfg):
    kind = cfg.get("type")
    if kind == "picamera":
        return PiCamSource(cfg)
    if kind == "usb":
        return UsbSource(cfg)
    if kind == "fake":
        return FakeSource(cfg)
    raise ValueError(f"unknown camera type: {kind!r} (use picamera, usb or fake)")
