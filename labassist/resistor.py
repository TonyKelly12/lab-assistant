"""Through-hole resistor colour-band decoder (classical CV).  SHARED (pure).

A deliberate companion to the vision-language model: VLMs read printed part
numbers well but are shaky at counting and naming resistor colour bands, so the
vision bridge runs this first and hands its hypothesis to the VLM as a
cross-check. Nothing here needs a GPU or a network.

Two layers, so the decode logic is testable without any image:

  decode_bands(names)        ordered band-colour names -> value/tolerance dict
  read_resistor(image_bgr)   crop of one resistor -> segment bands -> decode

`read_resistor` assumes a single, roughly horizontal resistor that fills most
of the crop (what you get holding one part up to the inspection camera).
"""
import numpy as np

try:
    import cv2
except Exception:  # noqa: BLE001 - keep decode_bands import-safe without OpenCV
    cv2 = None

# name -> (significant digit or None, multiplier power of ten, tolerance % or None)
COLOR_CODE = {
    "black":  (0,  0,  None),
    "brown":  (1,  1,  1.0),
    "red":    (2,  2,  2.0),
    "orange": (3,  3,  None),
    "yellow": (4,  4,  None),
    "green":  (5,  5,  0.5),
    "blue":   (6,  6,  0.25),
    "violet": (7,  7,  0.1),
    "grey":   (8,  8,  0.05),
    "white":  (9,  9,  None),
    "gold":   (None, -1, 5.0),
    "silver": (None, -2, 10.0),
}

# Reference colours as sRGB (R, G, B). Matching is done in CIELab for a more
# perceptual distance. gold/silver are deliberately warm/neutral-light.
_REF_RGB = {
    "black":  (30, 30, 30),
    "brown":  (105, 65, 45),
    "red":    (190, 35, 35),
    "orange": (225, 110, 30),
    "yellow": (225, 200, 50),
    "green":  (40, 140, 70),
    "blue":   (45, 70, 165),
    "violet": (135, 70, 170),
    "grey":   (135, 135, 135),
    "white":  (235, 235, 235),
    "gold":   (195, 155, 70),
    "silver": (190, 190, 190),
}

# Colours that are never a significant digit -> they anchor orientation and are
# the usual tolerance band.
_TOLERANCE_ONLY = {"gold", "silver"}


def _lab_refs():
    """Reference colours as Lab rows, built once. Requires OpenCV."""
    if cv2 is None:
        raise RuntimeError("OpenCV is required for colour classification")
    names = list(_REF_RGB)
    rgb = np.array([_REF_RGB[n] for n in names], np.uint8).reshape(-1, 1, 3)
    bgr = rgb[:, :, ::-1].copy()
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float32)
    return names, lab


def classify_color(bgr):
    """Nearest band colour name for one BGR pixel/mean, plus its Lab distance."""
    names, refs = _lab_refs()
    px = np.array([[list(bgr)]], np.uint8)
    lab = cv2.cvtColor(px, cv2.COLOR_BGR2Lab).reshape(3).astype(np.float32)
    d = np.linalg.norm(refs - lab, axis=1)
    i = int(np.argmin(d))
    return names[i], float(d[i])


def format_ohms(ohms):
    """Human reading: 1_000 -> '1 kΩ', 4_700_000 -> '4.7 MΩ'."""
    if ohms is None:
        return "unknown"
    for div, suffix in ((1e9, "GΩ"), (1e6, "MΩ"), (1e3, "kΩ")):
        if ohms >= div:
            v = ohms / div
            return f"{v:g} {suffix}"
    return f"{ohms:g} Ω"


def decode_bands(names):
    """Decode an ordered list of band-colour names into a value.

    Orientation: a gold/silver band is never a digit, so if it lands first the
    sequence is read backwards and we flip it. Returns a dict with ohms (or
    None if undecodable), tolerance, the (possibly flipped) band names, a
    human-readable string, and whether the orientation was resolved.
    """
    names = [str(n).lower() for n in names if n]
    n = len(names)
    out = {"bands": names, "ohms": None, "tolerance": None,
           "text": "undecodable", "oriented": False}
    if n < 3 or n > 6:
        return out

    # Resolve orientation from the tolerance-only anchor.
    first_tol = names[0] in _TOLERANCE_ONLY
    last_tol = names[-1] in _TOLERANCE_ONLY
    if first_tol and not last_tol:
        names = list(reversed(names))
        out["bands"] = names
        out["oriented"] = True
    elif last_tol and not first_tol:
        out["oriented"] = True

    if n in (4, 6):
        digit_names, mult_name, tol_name = names[:2], names[2], names[3]
    elif n == 5:
        digit_names, mult_name, tol_name = names[:3], names[3], names[4]
    else:  # n == 3: no tolerance band -> ±20%
        digit_names, mult_name, tol_name = names[:2], names[2], None

    digits = []
    for dn in digit_names:
        d = COLOR_CODE.get(dn, (None, None, None))[0]
        if d is None:
            return out  # a non-digit colour in a digit slot -> can't decode
        digits.append(str(d))
    mult = COLOR_CODE.get(mult_name, (None, None, None))[1]
    if mult is None:
        return out

    ohms = int("".join(digits)) * (10.0 ** mult)
    ohms = int(ohms) if ohms >= 1 else ohms
    tol = 20.0 if tol_name is None else COLOR_CODE.get(tol_name, (None, None, None))[2]

    out["ohms"] = ohms
    out["tolerance"] = tol
    tol_txt = f" ±{tol:g}%" if tol is not None else ""
    out["text"] = f"{format_ohms(ohms)}{tol_txt}"
    return out


def segment_bands(image_bgr, max_bands=6):
    """Find colour bands along a horizontal resistor crop.

    Takes a central horizontal strip, finds the resistor body colour (the most
    common column colour), then groups runs of columns that differ from the
    body into bands. Returns a list of {center, width, bgr} left to right.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is required for segmentation")
    img = image_bgr
    h, w = img.shape[:2]
    if h < 4 or w < 8:
        return []
    # Central strip avoids the rounded top/bottom edges and the leads.
    y0, y1 = int(h * 0.30), int(h * 0.70)
    strip = img[y0:y1 if y1 > y0 else h]
    col = np.median(strip, axis=0)                       # (w, 3) median colour per column
    blur = cv2.GaussianBlur(col.reshape(1, -1, 3).astype(np.float32), (1, 9), 0).reshape(-1, 3)

    # Body colour = the column colour that most columns sit near.
    #   distance of every column to a coarse palette of itself -> mode via median.
    body = np.median(blur, axis=0)
    dist = np.linalg.norm(blur - body, axis=1)
    # A column is part of a band when it sits well off the body colour.
    thresh = float(np.median(dist)) + 18.0
    is_band = dist > thresh

    bands, i = [], 0
    while i < w:
        if is_band[i]:
            j = i
            while j < w and is_band[j]:
                j += 1
            width = j - i
            if width >= max(2, w // 60):                 # ignore specks
                seg = blur[i:j]
                bands.append({"center": (i + j) // 2, "width": width,
                              "bgr": tuple(float(v) for v in np.median(seg, axis=0))})
            i = j
        else:
            i += 1

    # Merge the brightest/widest if we overshoot the plausible band count.
    bands.sort(key=lambda b: b["center"])
    return bands[:max_bands]


def read_resistor(image_bgr):
    """Full pipeline: crop of one resistor -> decoded value + confidence.

    Confidence is a rough 0..1 from how cleanly each band colour matched a
    reference (large Lab distances -> low confidence). This is a hint for the
    VLM cross-check, not a guarantee.
    """
    bands = segment_bands(image_bgr)
    if len(bands) < 3:
        return {"ohms": None, "tolerance": None, "bands": [], "confidence": 0.0,
                "text": "no clear bands", "hypothesis": None}
    names, dists = [], []
    for b in bands:
        name, dist = classify_color(b["bgr"])
        names.append(name)
        dists.append(dist)
    decoded = decode_bands(names)
    # Map mean Lab distance (~0 perfect, ~60 poor) to a confidence.
    mean_dist = float(np.mean(dists)) if dists else 99.0
    conf = max(0.0, min(1.0, 1.0 - mean_dist / 60.0))
    if decoded["ohms"] is None:
        conf *= 0.3
    return {
        "ohms": decoded["ohms"],
        "tolerance": decoded["tolerance"],
        "bands": decoded["bands"],
        "confidence": round(conf, 2),
        "text": decoded["text"],
        "hypothesis": hypothesis(decoded, conf),
    }


def hypothesis(decoded, confidence):
    """One short line to hand the VLM as a cross-check, or None if hopeless."""
    if not decoded or decoded.get("ohms") is None:
        return None
    bands = ", ".join(decoded["bands"])
    return (f"A colour-band reader thinks this resistor is {decoded['text']} "
            f"(bands: {bands}; confidence {confidence:.0%}). "
            f"Confirm or correct from the image.")
