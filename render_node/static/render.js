// Projector page. Receives state from the render server over Server-Sent Events.
//
// Scene items are in bench millimetres. The whole <svg> is laid out with
// 1 CSS px = 1 mm, then warped onto the projector with a CSS matrix3d built
// from the calibration homography (bench mm -> projector px).
"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const patternEl = document.getElementById("pattern");
const benchEl = document.getElementById("bench");
const statusEl = document.getElementById("status");

function matrix3d(H) {
  const h = H.map(r => r.map(v => v / H[2][2]));
  // CSS matrix3d is column-major.
  return `matrix3d(${[
    h[0][0], h[1][0], 0, h[2][0],
    h[0][1], h[1][1], 0, h[2][1],
    0, 0, 1, 0,
    h[0][2], h[1][2], 0, h[2][2],
  ].join(",")})`;
}

function el(tag, attrs, parent) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== undefined && v !== null) node.setAttribute(k, v);
  }
  parent.appendChild(node);
  return node;
}

function style(item, defaults) {
  return {
    stroke: item.stroke ?? defaults.stroke ?? "#00e5ff",
    fill: item.fill ?? defaults.fill ?? "none",
    "stroke-width": item.width ?? defaults.width ?? 1.5,
    opacity: item.opacity,
  };
}

// Each item type takes millimetre coordinates on the bench.
const DRAW = {
  rect: (it, g) => el("rect", { x: it.x, y: it.y, width: it.w, height: it.h, rx: it.r, ...style(it, {}) }, g),
  circle: (it, g) => el("circle", { cx: it.cx, cy: it.cy, r: it.r, ...style(it, {}) }, g),
  line: (it, g) => el("line", { x1: it.x1, y1: it.y1, x2: it.x2, y2: it.y2, ...style(it, {}) }, g),
  polyline: (it, g) => el("polyline", { points: it.points.map(p => p.join(",")).join(" "), ...style(it, {}) }, g),
  polygon: (it, g) => el("polygon", { points: it.points.map(p => p.join(",")).join(" "), ...style(it, {}) }, g),
  text: (it, g) => {
    const t = el("text", {
      x: it.x, y: it.y, "font-size": it.size ?? 8, fill: it.color ?? "#ffffff",
      "text-anchor": it.anchor ?? "start", "dominant-baseline": it.baseline ?? "auto",
      "font-weight": it.bold ? "700" : "400",
    }, g);
    t.textContent = it.text ?? "";
    return t;
  },
  image: (it, g) => el("image", { href: it.href, x: it.x, y: it.y, width: it.w, height: it.h, preserveAspectRatio: "xMidYMid meet" }, g),
  // Grid of dots or lines at a pitch, e.g. breadboard holes (pitch 2.54).
  grid: (it, g) => {
    const pitch = it.pitch ?? 2.54;
    const cols = Math.round(it.w / pitch), rows = Math.round(it.h / pitch);
    if (it.dots) {
      for (let r = 0; r <= rows; r++) for (let c = 0; c <= cols; c++) {
        el("circle", { cx: it.x + c * pitch, cy: it.y + r * pitch, r: it.dot ?? pitch * 0.25, fill: it.stroke ?? "#00e5ff" }, g);
      }
    } else {
      const s = style(it, { width: 0.4 });
      for (let c = 0; c <= cols; c++) el("line", { x1: it.x + c * pitch, y1: it.y, x2: it.x + c * pitch, y2: it.y + it.h, ...s }, g);
      for (let r = 0; r <= rows; r++) el("line", { x1: it.x, y1: it.y + r * pitch, x2: it.x + it.w, y2: it.y + r * pitch, ...s }, g);
    }
  },
  cross: (it, g) => {
    const s = it.size ?? 10, st = style(it, {});
    el("line", { x1: it.x - s, y1: it.y, x2: it.x + s, y2: it.y, ...st }, g);
    el("line", { x1: it.x, y1: it.y - s, x2: it.x, y2: it.y + s, ...st }, g);
  },
};

function showPattern(state) {
  benchEl.style.display = "none";
  patternEl.style.display = "block";
  const [w, h] = state.projector_size;
  const p = state.pattern || { kind: "black" };
  patternEl.innerHTML = "";
  patternEl.style.width = w + "px";
  patternEl.style.height = h + "px";
  if (p.kind === "image") {
    const img = document.createElement("img");
    img.src = p.url;
    img.width = w;
    img.height = h;
    patternEl.appendChild(img);
    patternEl.style.background = "#000";
  } else {
    patternEl.style.background = p.kind === "white" ? "#fff" : "#000";
  }
  statusEl.style.display = "none";
}

function showScene(state) {
  patternEl.style.display = "none";
  benchEl.style.display = "block";
  benchEl.innerHTML = "";
  const cal = state.calibration;
  let x0, y0, W, D;
  if (cal) {
    const [ax0, ay0, ax1, ay1] = cal.area_mm;
    [x0, y0, W, D] = [ax0, ay0, ax1 - ax0, ay1 - ay0];
    // The svg's local px (u, v) are bench mm shifted by the area origin: H @ translate(x0, y0).
    const H = cal.H_bench_to_proj;
    const Hs = H.map(r => [r[0], r[1], r[0] * x0 + r[1] * y0 + r[2]]);
    benchEl.style.transform = matrix3d(Hs);
    statusEl.style.display = "none";
  } else {
    // Not calibrated: fit a 1000 x 600 mm area into the screen so test scenes still show.
    [x0, y0, W, D] = [0, 0, 1000, 600];
    const [pw, ph] = state.projector_size;
    const s = Math.min(pw / W, ph / D) * 0.9;
    benchEl.style.transform = `translate(${(pw - W * s) / 2}px, ${(ph - D * s) / 2}px) scale(${s})`;
    statusEl.textContent = "not calibrated: run calibration/projector_calibrate.py";
    statusEl.style.display = "block";
    el("rect", { x: 0, y: 0, width: W, height: D, fill: "none", stroke: "#333", "stroke-width": 2 }, benchEl);
  }
  benchEl.setAttribute("width", W);
  benchEl.setAttribute("height", D);
  benchEl.setAttribute("viewBox", `${x0} ${y0} ${W} ${D}`);
  benchEl.style.width = W + "px";
  benchEl.style.height = D + "px";
  for (const item of (state.scene && state.scene.items) || []) {
    const draw = DRAW[item.type];
    if (draw) {
      try { draw(item, benchEl); } catch (e) { console.error("bad item", item, e); }
    } else {
      console.warn("unknown item type", item.type);
    }
  }
}

function apply(state) {
  if (state.mode === "pattern") showPattern(state);
  else showScene(state);
}

function connect() {
  const es = new EventSource("/events");
  es.onmessage = ev => apply(JSON.parse(ev.data));
  es.onerror = () => {
    statusEl.textContent = "reconnecting to render server…";
    statusEl.style.display = "block";
  };
}

connect();
