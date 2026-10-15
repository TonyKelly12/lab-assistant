"""Make the four bench markers and a checkerboard for lens calibration.

Two formats:
  *.png  print at 100% / "actual size" (turn off "fit to page"), then measure
         the 100 mm scale bar with a ruler to confirm the printer didn't rescale
  *.svg  exact-size vectors for a laser engraver (millimetre units): black
         fill = engrave, red outline = cut. Engrave dark on a light material
         (plywood, birch, cardstock, white acrylic, anodised aluminium).

Run:  python3 calibration/generate_markers.py --out prints/
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labassist import aruco  # noqa: E402

DPI = 300
PX_PER_MM = DPI / 25.4
LETTER_MM = (215.9, 279.4)
PLACEMENT = {0: "BACK LEFT", 1: "BACK RIGHT", 2: "FRONT RIGHT", 3: "FRONT LEFT"}


def mm(v):
    return int(round(v * PX_PER_MM))


def scale_bar(img, x, y):
    cv2.rectangle(img, (mm(x), mm(y)), (mm(x + 100), mm(y + 3)), 0, -1)
    for t in range(0, 101, 10):
        cv2.line(img, (mm(x + t), mm(y - 2)), (mm(x + t), mm(y + 3)), 0, 3)
    cv2.putText(img, "100 mm - measure me", (mm(x), mm(y + 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3)


def bench_sheet(marker_mm):
    w, h = LETTER_MM
    img = np.full((mm(h), mm(w)), 255, np.uint8)
    cell_w, cell_h = w / 2, (h - 30) / 2
    for i, marker_id in enumerate(sorted(PLACEMENT)):
        cx = cell_w * (i % 2) + cell_w / 2
        cy = 15 + cell_h * (i // 2) + cell_h / 2
        x0, y0 = cx - marker_mm / 2, cy - marker_mm / 2
        side = mm(marker_mm)
        img[mm(y0):mm(y0) + side, mm(x0):mm(x0) + side] = aruco.marker_image(marker_id, side)
        label = f"ID {marker_id}: {PLACEMENT[marker_id]}"
        cv2.putText(img, label, (mm(x0), mm(y0 + marker_mm + 9)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)
    scale_bar(img, 15, h - 14)
    return img


def checkerboard(cols, rows, square_mm):
    """cols x rows squares (inner corners: cols-1 x rows-1), landscape letter."""
    w, h = LETTER_MM[1], LETTER_MM[0]
    img = np.full((mm(h), mm(w)), 255, np.uint8)
    bw, bh = cols * square_mm, rows * square_mm
    if bw > w - 20 or bh > h - 30:
        raise SystemExit("checkerboard too big for the page; use smaller --square-mm")
    x0, y0 = (w - bw) / 2, 10
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                cv2.rectangle(img, (mm(x0 + c * square_mm), mm(y0 + r * square_mm)),
                              (mm(x0 + (c + 1) * square_mm) - 1, mm(y0 + (r + 1) * square_mm) - 1), 0, -1)
    scale_bar(img, x0, h - 12)
    cv2.putText(img, f"{cols - 1}x{rows - 1} inner corners, {square_mm} mm squares",
                (mm(x0 + 110), mm(h - 9)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)
    return img


def marker_cells(marker_id):
    """Marker as a grid of cells (True = black), including its black border."""
    cells = 4 + 2                                   # 4x4 data bits + 1-cell border each side
    img = aruco.marker_image(marker_id, cells * 20)
    return [[img[r * 20 + 10, c * 20 + 10] < 128 for c in range(cells)] for r in range(cells)]


def _rects_path(cells, x0, y0, cell):
    """One SVG path of filled squares, merging horizontal runs of black cells."""
    parts = []
    for r, row in enumerate(cells):
        c = 0
        while c < len(row):
            if row[c]:
                start = c
                while c < len(row) and row[c]:
                    c += 1
                x, y, w = x0 + start * cell, y0 + r * cell, (c - start) * cell
                parts.append(f"M{x:.3f} {y:.3f}h{w:.3f}v{cell:.3f}h{-w:.3f}z")
            else:
                c += 1
    return "".join(parts)


def bench_svg(marker_mm):
    """Four markers on separate tiles, each with a white quiet zone and a cut outline."""
    cell = marker_mm / 6
    quiet = cell                                    # white margin around the marker
    label_h = max(8.0, marker_mm * 0.15)
    tile_w = marker_mm + 2 * quiet
    tile_h = marker_mm + 2 * quiet + label_h
    gap = 6.0
    width = 4 * tile_w + 5 * gap
    height = tile_h + 2 * gap
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.2f}mm" height="{height:.2f}mm" '
           f'viewBox="0 0 {width:.3f} {height:.3f}">',
           f'<!-- ArUco {aruco.DICT_NAME} bench markers, {marker_mm:g} mm. Black = engrave, red = cut. -->']
    for i, marker_id in enumerate(sorted(PLACEMENT)):
        tx, ty = gap + i * (tile_w + gap), gap
        out.append(f'<rect x="{tx:.3f}" y="{ty:.3f}" width="{tile_w:.3f}" height="{tile_h:.3f}" '
                   f'fill="none" stroke="#ff0000" stroke-width="0.1"/>')
        out.append(f'<path d="{_rects_path(marker_cells(marker_id), tx + quiet, ty + quiet, cell)}" fill="#000000"/>')
        out.append(f'<text x="{tx + tile_w / 2:.3f}" y="{ty + 2 * quiet + marker_mm + label_h * 0.7:.3f}" '
                   f'font-family="Arial, sans-serif" font-size="{label_h * 0.55:.2f}" text-anchor="middle" '
                   f'fill="#000000">ID {marker_id} {PLACEMENT[marker_id]}</text>')
    out.append("</svg>")
    return "\n".join(out)


def checkerboard_svg(cols, rows, square_mm):
    """cols x rows squares (inner corners: cols-1 x rows-1) with a cut outline."""
    margin = square_mm
    width, height = cols * square_mm + 2 * margin, rows * square_mm + 2 * margin
    cells = [[(r + c) % 2 == 0 for c in range(cols)] for r in range(rows)]
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.2f}mm" height="{height:.2f}mm" '
        f'viewBox="0 0 {width:.3f} {height:.3f}">',
        f'<!-- {cols - 1}x{rows - 1} inner corners, {square_mm:g} mm squares. Black = engrave, red = cut. -->',
        f'<rect x="0" y="0" width="{width:.3f}" height="{height:.3f}" fill="none" stroke="#ff0000" stroke-width="0.1"/>',
        f'<path d="{_rects_path(cells, margin, margin, square_mm)}" fill="#000000"/>',
        "</svg>"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="prints")
    p.add_argument("--marker-mm", type=float, default=60)
    p.add_argument("--square-mm", type=float, default=25)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, img in (("bench-markers.png", bench_sheet(a.marker_mm)),
                      ("checkerboard-9x6.png", checkerboard(10, 7, a.square_mm))):
        path = os.path.join(a.out, name)
        cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        # Embed DPI so print dialogs default to the right physical size.
        try:
            from PIL import Image
            Image.open(path).save(path, dpi=(DPI, DPI))
        except ImportError:
            pass
        print("wrote", path)
    for name, svg in (("bench-markers.svg", bench_svg(a.marker_mm)),
                      ("checkerboard-9x6.svg", checkerboard_svg(10, 7, a.square_mm))):
        path = os.path.join(a.out, name)
        with open(path, "w") as f:
            f.write(svg)
        print("wrote", path)
    print("PNG: print at 100% scale (actual size), then check the 100 mm bar with a ruler.")
    print("SVG: laser engrave the black, cut the red outlines; sizes are exact millimetres.")


if __name__ == "__main__":
    main()
