// ==================== UNIVERSAL SMART VISION: TRAFFIC & ANPR ENGINE ====================
let trfVideo = null;
let trfCanvas = null;
let trfCtx = null;
let trfFrameCount = 0;
let trfLastFpsTime = performance.now();

const SAMPLE_PLATES = [
  { plate: "MH-09-EV-2026", type: "Sedan", speed: "52 km/h", conf: "98.4%" },
  { plate: "MH-14-GH-1008", type: "SUV", speed: "61 km/h", conf: "97.1%" },
  { plate: "DL-01-AX-9932", type: "Electric Cart", speed: "28 km/h", conf: "99.0%" },
  { plate: "KA-04-TT-4455", type: "Heavy Truck", speed: "41 km/h", conf: "95.8%" }
];
let sampleIdx = 0;

function loadTrafficClip() {
  trfCanvas = document.getElementById("trfCanvas");
  if (!trfCanvas) return;
  trfCtx = trfCanvas.getContext("2d");
  trfVideo = document.getElementById("trfVideo");
  trfVideo.src = "/static/videos/cv_demo_04.mp4";
  trfVideo.loop = true;
  trfVideo.play().then(() => {
    requestAnimationFrame(renderTrafficLoop);
  }).catch(() => {
    requestAnimationFrame(renderTrafficLoop);
  });
}

function renderTrafficLoop() {
  if (!trfCanvas || !trfCtx) return;
  const w = trfCanvas.width;
  const h = trfCanvas.height;

  if (trfVideo && trfVideo.readyState >= 2) {
    trfCtx.drawImage(trfVideo, 0, 0, w, h);
  } else {
    trfCtx.fillStyle = "#0a0c14";
    trfCtx.fillRect(0, 0, w, h);
    // Draw road lanes
    trfCtx.strokeStyle = "#27272a";
    trfCtx.lineWidth = 3;
    trfCtx.beginPath();
    trfCtx.moveTo(w * 0.2, h); trfCtx.lineTo(w * 0.45, h * 0.3);
    trfCtx.moveTo(w * 0.8, h); trfCtx.lineTo(w * 0.55, h * 0.3);
    trfCtx.stroke();
  }

  // Draw Vehicle Bounding Boxes
  const vx = w * 0.38, vy = h * 0.45, vw = 140, vh = 95;
  trfCtx.strokeStyle = "#38bdf8";
  trfCtx.lineWidth = 2;
  trfCtx.strokeRect(vx, vy, vw, vh);

  // Label
  trfCtx.fillStyle = "rgba(56, 189, 248, 0.85)";
  trfCtx.fillRect(vx, vy - 18, 110, 18);
  trfCtx.fillStyle = "#000";
  trfCtx.font = "bold 10px JetBrains Mono, monospace";
  trfCtx.fillText("CAR #14 [52 km/h]", vx + 4, vy - 5);

  // Number Plate Sub-Box
  const px = vx + 35, py = vy + vh - 22, pw = 70, ph = 18;
  trfCtx.strokeStyle = "#facc15";
  trfCtx.lineWidth = 2;
  trfCtx.strokeRect(px, py, pw, ph);

  // FPS
  trfFrameCount++;
  const now = performance.now();
  if (now - trfLastFpsTime >= 1000) {
    const fps = Math.round((trfFrameCount * 1000) / (now - trfLastFpsTime));
    const fpsElem = document.getElementById("trfFps");
    if (fpsElem) fpsElem.textContent = `${fps} FPS`;
    trfFrameCount = 0;
    trfLastFpsTime = now;
  }

  requestAnimationFrame(renderTrafficLoop);
}

function simulateVehicleCapture() {
  sampleIdx = (sampleIdx + 1) % SAMPLE_PLATES.length;
  const p = SAMPLE_PLATES[sampleIdx];

  const plateElem = document.getElementById("anprPlateNumber");
  const confElem = document.getElementById("anprConfVal");
  const speedElem = document.getElementById("anprSpeedVal");
  const log = document.getElementById("plateLogContainer");

  if (plateElem) plateElem.textContent = p.plate;
  if (confElem) confElem.textContent = p.conf;
  if (speedElem) speedElem.textContent = p.speed;

  if (log) {
    const item = document.createElement("div");
    item.style.cssText = "display: flex; justify-content: space-between; align-items: center; background: rgba(56, 189, 248, 0.08); padding: 6px 10px; border-radius: 4px; border-left: 3px solid #38bdf8; font-size: 11px;";
    item.innerHTML = `<div><code style="color: #38bdf8; font-weight: 700;">${p.plate}</code> &bull; ${p.type}</div><span style="color: var(--text-dim); font-size: 10px;">Just now &bull; ${p.speed}</span>`;
    log.insertBefore(item, log.firstChild);
    if (log.children.length > 5) log.removeChild(log.lastChild);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  loadTrafficClip();
});
