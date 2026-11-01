"""Resistor colour-band decoder tests.

The decode core (band names -> value) is tested without any image. A couple of
synthetic resistor images check the end-to-end segment-and-decode path.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labassist import resistor as R  # noqa: E402

cv2 = pytest.importorskip("cv2")


# -- the decode core -------------------------------------------------------

def test_decode_four_band():
    d = R.decode_bands(["brown", "black", "red", "gold"])
    assert d["ohms"] == 1000 and d["tolerance"] == 5.0
    assert d["text"] == "1 kΩ ±5%"


def test_decode_five_band():
    d = R.decode_bands(["brown", "black", "black", "red", "brown"])
    assert d["ohms"] == 10000 and d["tolerance"] == 1.0


def test_decode_three_band_is_twenty_percent():
    d = R.decode_bands(["brown", "black", "red"])
    assert d["ohms"] == 1000 and d["tolerance"] == 20.0


def test_decode_flips_when_tolerance_band_is_first():
    forward = R.decode_bands(["brown", "red", "red", "gold"])
    flipped = R.decode_bands(["gold", "red", "red", "brown"])
    assert flipped["oriented"] and flipped["ohms"] == forward["ohms"] == 1200


def test_decode_rejects_nonsense():
    assert R.decode_bands(["gold", "gold"])["ohms"] is None          # too few
    assert R.decode_bands(["gold", "black", "red", "gold"])["ohms"] is None  # gold in a digit slot


def test_format_ohms():
    assert R.format_ohms(4700) == "4.7 kΩ"
    assert R.format_ohms(1_000_000) == "1 MΩ"
    assert R.format_ohms(220) == "220 Ω"


# -- the image pipeline ----------------------------------------------------

_BGR = {
    "brown": (45, 65, 105), "black": (30, 30, 30), "red": (35, 35, 190),
    "orange": (30, 110, 225), "yellow": (50, 200, 225), "green": (70, 140, 40),
    "blue": (165, 70, 45), "violet": (170, 70, 135), "grey": (135, 135, 135),
    "white": (235, 235, 235), "gold": (70, 155, 195), "silver": (190, 190, 190),
}


def _resistor_image(bands, body=(150, 200, 210), w=400, h=120):
    """Draw a horizontal resistor: tan body with evenly spaced colour bands."""
    img = np.full((h, w, 3), body, np.uint8)
    n = len(bands)
    xs = np.linspace(w * 0.18, w * 0.78, n).astype(int)
    for x, name in zip(xs, bands):
        cv2.rectangle(img, (x - 14, 12), (x + 14, h - 12), _BGR[name], -1)
    return img


def test_read_resistor_synthetic_1k():
    res = R.read_resistor(_resistor_image(["brown", "black", "red", "gold"]))
    assert res["ohms"] == 1000
    assert res["tolerance"] == 5.0
    assert res["confidence"] > 0.5
    assert res["hypothesis"] and "1 kΩ" in res["hypothesis"]


def test_read_resistor_synthetic_47k():
    res = R.read_resistor(_resistor_image(["yellow", "violet", "orange", "gold"]))
    assert res["ohms"] == 47000


def test_read_resistor_no_bands_is_safe():
    blank = np.full((120, 400, 3), (150, 200, 210), np.uint8)
    res = R.read_resistor(blank)
    assert res["ohms"] is None and res["confidence"] == 0.0
