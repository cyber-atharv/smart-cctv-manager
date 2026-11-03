// ==================== UNIVERSAL SMART VISION: INDUSTRIAL SAFETY ENGINE ====================
let indStream = null;
let indVideo = null;
let indCanvas = null;
let indCtx = null;

let hasHelmetViolation = false;
let hasVestViolation = false;
let hasFireAnomaly = false;

async function startIndustrialCam() {
  indVideo = document.getElementById("indVideo");
  indCanvas = document.getElementById("indCanvas");
  if (!indCanvas) return;
  indCtx = indCanvas.getContext("2d");

  try {
    indStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60, min: 30 } },
      audio: false
    });
    indVideo.srcObject = indStream;
    await indVideo.play();
    requestAnimationFrame(renderIndustrialLoop);
  } catch (err) {
    loadIndustrialClip();
  }
}

function loadIndustrialClip() {
  indCanvas = document.getElementById("indCanvas");
  if (!indCanvas) return;
  indCtx = indCanvas.getContext("2d");
  indVideo = document.getElementById("indVideo");
  indVideo.srcObject = null;
  indVideo.src = "/static/videos/cv_demo_06.mp4";
  indVideo.loop = true;
  indVideo.play().then(() => {
    requestAnimationFrame(renderIndustrialLoop);
  }).catch(() => {
    requestAnimationFrame(renderIndustrialLoop);
  });
}

let indFrameCount = 0;
let indLastFpsTime = performance.now();

function renderIndustrialLoop() {
  if (!indCanvas || !indCtx) return;
  const w = indCanvas.width;
  const h = indCanvas.height;

  if (indVideo && indVideo.readyState >= 2) {
    indCtx.drawImage(indVideo, 0, 0, w, h);
  } else {
    indCtx.fillStyle = "#10121a";
    indCtx.fillRect(0, 0, w, h);
  }

  // Draw Machine Exclusion Zone
  const zx = w * 0.65, zy = h * 0.40, zw = w * 0.30, zh = h * 0.50;
  indCtx.save();
  indCtx.fillStyle = "rgba(245, 158, 11, 0.08)";
  indCtx.fillRect(zx, zy, zw, zh);
  indCtx.strokeStyle = "rgba(245, 158, 11, 0.6)";
  indCtx.lineWidth = 2;
  indCtx.strokeRect(zx, zy, zw, zh);
  indCtx.fillStyle = "#f59e0b";
  indCtx.font = "bold 10px JetBrains Mono, monospace";
  indCtx.fillText("RESTRICTED: MACHINE BAY B", zx + 8, zy + 18);
  indCtx.restore();

  // Draw Worker Bounding Box
  const px = w * 0.30, py = h * 0.25, pw = 100, ph = 210;
  const isViolation = hasHelmetViolation || hasVestViolation;

  indCtx.strokeStyle = isViolation ? "#ef4444" : "#22c55e";
  indCtx.lineWidth = 2;
  indCtx.strokeRect(px, py, pw, ph);

  // Helmet Box
  const hx = px + 15, hy = py - 18, hw = 70, hh = 26;
  indCtx.fillStyle = hasHelmetViolation ? "rgba(239, 68, 68, 0.85)" : "rgba(34, 197, 94, 0.85)";
  indCtx.fillRect(hx, hy, hw, hh);
  indCtx.fillStyle = "#ffffff";
  indCtx.font = "bold 9px JetBrains Mono, monospace";
  indCtx.fillText(hasHelmetViolation ? "NO HELMET!" : "HELMET 96%", hx + 4, hy + 17);

  // Vest Box
  const vx = px + 10, vy = py + 30, vw = 80, vh = 65;
  indCtx.strokeStyle = hasVestViolation ? "#ef4444" : "#22c55e";
  indCtx.lineWidth = 2;
  indCtx.strokeRect(vx, vy, vw, vh);
  indCtx.fillStyle = hasVestViolation ? "#ef4444" : "#22c55e";
  indCtx.font = "bold 9px JetBrains Mono, monospace";
  indCtx.fillText(hasVestViolation ? "VEST MISSING!" : "VEST 92%", vx + 2, vy - 4);

  // Fire Anomaly Simulation Overlay
  if (hasFireAnomaly) {
    const fx = w * 0.70, fy = h * 0.25;
    const grad = indCtx.createRadialGradient(fx, fy, 10, fx, fy, 80);
    grad.addColorStop(0, "rgba(239, 68, 68, 0.8)");
    grad.addColorStop(0.5, "rgba(245, 158, 11, 0.5)");
    grad.addColorStop(1, "rgba(0, 0, 0, 0)");
    indCtx.fillStyle = grad;
    indCtx.beginPath();
    indCtx.arc(fx, fy, 80, 0, Math.PI * 2);
    indCtx.fill();
  }

  // FPS
  indFrameCount++;
  const now = performance.now();
  if (now - indLastFpsTime >= 1000) {
    const fps = Math.round((indFrameCount * 1000) / (now - indLastFpsTime));
    const fpsElem = document.getElementById("indFps");
    if (fpsElem) fpsElem.textContent = `${fps} FPS`;
    indFrameCount = 0;
    indLastFpsTime = now;
  }

  requestAnimationFrame(renderIndustrialLoop);
}

function togglePPEViolation(item) {
  if (item === 'helmet') hasHelmetViolation = !hasHelmetViolation;
  if (item === 'vest') hasVestViolation = !hasVestViolation;
  updateIndustrialUI();
}

function toggleFireAnomaly() {
  hasFireAnomaly = !hasFireAnomaly;
  const banner = document.getElementById("fireAlertBanner");
  if (banner) banner.style.display = hasFireAnomaly ? "block" : "none";
}

function updateIndustrialUI() {
  const badge = document.getElementById("ppeStatusBadge");
  const hLabel = document.getElementById("helmetConfLabel");
  const hBar = document.getElementById("helmetConfBar");
  const vLabel = document.getElementById("vestConfLabel");
  const vBar = document.getElementById("vestConfBar");
  const rate = document.getElementById("complianceRateVal");

  if (hasHelmetViolation) {
    hLabel.textContent = "VIOLATION: MISSING (12%)";
    hLabel.style.color = "var(--danger)";
    hBar.style.width = "12%";
    hBar.style.background = "var(--danger)";
  } else {
    hLabel.textContent = "VERIFIED (96%)";
    hLabel.style.color = "var(--success)";
    hBar.style.width = "96%";
    hBar.style.background = "var(--success)";
  }

  if (hasVestViolation) {
    vLabel.textContent = "VIOLATION: MISSING (08%)";
    vLabel.style.color = "var(--danger)";
    vBar.style.width = "8%";
    vBar.style.background = "var(--danger)";
  } else {
    vLabel.textContent = "VERIFIED (92%)";
    vLabel.style.color = "var(--success)";
    vBar.style.width = "92%";
    vBar.style.background = "var(--success)";
  }

  if (hasHelmetViolation || hasVestViolation) {
    badge.textContent = "⚠ PPE NON-COMPLIANCE";
    badge.style.background = "rgba(239, 68, 68, 0.2)";
    badge.style.color = "var(--danger)";
    if (rate) rate.textContent = "50%";
  } else {
    badge.textContent = "✓ FULLY COMPLIANT";
    badge.style.background = "rgba(34, 197, 94, 0.15)";
    badge.style.color = "var(--success)";
    if (rate) rate.textContent = "100%";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  startIndustrialCam();
});
