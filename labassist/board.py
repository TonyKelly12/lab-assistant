"""Find breadboards (bright rectangles) on the bench in a camera image.

Works in bench millimetres using the calibration's camera->bench mapping,
so sizes and positions come out in real units without any marker on the
board itself.
"""
import cv2
import numpy as np

from .geometry import apply, undistort_points

HALF_BREADBOARD_MM = (55.0, 83.0)   # 400-point "half size"; full size is about 55 x 165


def bench_mask(shape, H_cam_to_bench, area_mm, exclude_mm=(), exclude_radius=75.0):
    """Camera-pixel mask of the bench area, with circles cut out around the
    printed bench markers (their wooden plates are bright and board-sized)."""
    H_bench_to_cam = np.linalg.inv(H_cam_to_bench)
    x0, y0, x1, y1 = area_mm
    quad = apply(H_bench_to_cam, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).astype(np.int32)
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, quad, 255)
    for cx, cy in exclude_mm:
        ring = [[cx + exclude_radius * np.cos(a), cy + exclude_radius * np.sin(a)]
                for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
        cv2.fillConvexPoly(mask, apply(H_bench_to_cam, ring).astype(np.int32), 0)
    return mask


def local_whiteness(image, mask=None):
    """How much whiter each pixel is than the bench around it.

    Brightness minus a heavily blurred copy of the bench (so the projector's
    bright centre and dim edges cancel out). Only bench pixels feed the
    blur, so the dark surroundings don't make the bench's own edges look
    bright. When the image is well lit, colourfulness is subtracted too
    (white plastic vs brown MDF). Returns float32, same size as the image.
    """
    if image.ndim == 2:
        L = image.astype(np.float32)
        chroma = np.zeros_like(L)
    else:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[..., 0]
        chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    m = np.ones_like(L) if mask is None else (mask > 0).astype(np.float32)
    f = 0.125
    small_L = cv2.resize(L * m, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    small_m = cv2.resize(m, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    blur_L = cv2.GaussianBlur(small_L, (0, 0), 20)
    blur_m = cv2.GaussianBlur(small_m, (0, 0), 20)
    background = cv2.resize(blur_L / np.maximum(blur_m, 1e-3), (L.shape[1], L.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
    score = L - background
    bench_L = L[m > 0] if m.any() else L
    if np.median(bench_L) > 80:           # colour is only trustworthy on a well-lit bench
        score -= 1.5 * cv2.GaussianBlur(chroma, (0, 0), 3)
    score[m == 0] = 0
    return score


def find_boards(gray, H_cam_to_bench, mask, intrinsics=None, min_mm=20.0, max_mm=250.0, min_fill=0.75):
    """Return candidate boards, biggest first.

    Each: {"center": [x, y], "size": [short, long], "angle": deg of the long
    side from the bench x axis (0-180), "corners": 4x[x, y] (all bench mm),
    "fill": how solid the shape is, "px": corners in camera pixels}.
    """
    score = local_whiteness(gray, mask)
    vals = score[mask > 0]
    if vals.size == 0:
        return [], 0
    med, top = np.percentile(vals, [50, 99.9])
    thresh = med + 0.5 * (top - med)       # halfway between plain bench and the whitest things on it
    binary = ((score > thresh) & (mask > 0)).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for c in contours:
        area_px = cv2.contourArea(c)
        if area_px < 200:
            continue
        rect = cv2.minAreaRect(c)
        fill = area_px / max(rect[1][0] * rect[1][1], 1e-6)
        box = cv2.boxPoints(rect).astype(np.float32)
        mm = apply(H_cam_to_bench, undistort_points(box, intrinsics))
        sides = [np.linalg.norm(mm[(i + 1) % 4] - mm[i]) for i in range(4)]
        short, long_ = sorted([(sides[0] + sides[2]) / 2, (sides[1] + sides[3]) / 2])
        if fill < min_fill or short < min_mm or long_ > max_mm:
            continue
        i = 0 if (sides[0] + sides[2]) >= (sides[1] + sides[3]) else 1
        v = mm[i + 1] - mm[i]
        angle = float(np.degrees(np.arctan2(v[1], v[0])) % 180.0)
        found.append({
            "center": [round(float(v_), 1) for v_ in mm.mean(axis=0)],
            "size": [round(float(short), 1), round(float(long_), 1)],
            "angle": round(angle, 1),
            "corners": [[round(float(a), 1), round(float(b), 1)] for a, b in mm],
            "fill": round(float(fill), 2),
            "px": box.round(1).tolist(),
            "area_mm2": float(short * long_),
        })
    found.sort(key=lambda b: -b["area_mm2"])
    return found, float(thresh)


def outline_scene(boards, color="#00e5ff"):
    """Scene items that draw each found board's outline, centre and size."""
    items = []
    for n, b in enumerate(boards):
        items.append({"type": "polygon", "points": b["corners"], "stroke": color, "width": 0.8, "fill": "none"})
        items.append({"type": "cross", "x": b["center"][0], "y": b["center"][1], "size": 8, "stroke": color, "width": 0.6})
        top = min(b["corners"], key=lambda p: p[1])
        items.append({"type": "text", "x": top[0], "y": top[1] - 4, "size": 5, "color": color,
                      "text": f"#{n + 1} {b['size'][0]:.0f} x {b['size'][1]:.0f} mm  @ {b['angle']:.0f}°"})
    return {"items": items}
