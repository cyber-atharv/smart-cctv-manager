// ==================== UNIVERSAL SMART VISION: RETAIL & FOOTFALL ENGINE ====================
let retVideo = null;
let retCanvas = null;
let retCtx = null;
let retFrameCount = 0;
let retLastFpsTime = performance.now();

let countIn = 248;
let countOut = 214;

function loadRetailClip() {
  retCanvas = document.getElementById("retCanvas");
  if (!retCanvas) return;
  retCtx = retCanvas.getContext("2d");
  retVideo = document.getElementById("retVideo");
  retVideo.src = "/static/videos/cv_demo_02.mp4";
  retVideo.loop = true;
  retVideo.play().then(() => {
    requestAnimationFrame(renderRetailLoop);
  }).catch(() => {
    requestAnimationFrame(renderRetailLoop);
  });
}

function renderRetailLoop() {
  if (!retCanvas || !retCtx) return;
  const w = retCanvas.width;
  const h = retCanvas.height;

  if (retVideo && retVideo.readyState >= 2) {
    retCtx.drawImage(retVideo, 0, 0, w, h);
  } else {
    retCtx.fillStyle = "#0c0d14";
    retCtx.fillRect(0, 0, w, h);
  }

  // Draw Entry / Exit Tripwire
  retCtx.save();
  retCtx.strokeStyle = "rgba(168, 85, 247, 0.8)";
  retCtx.lineWidth = 3;
  retCtx.setLineDash([6, 6]);
  retCtx.beginPath();
  retCtx.moveTo(w * 0.15, h * 0.55);
  retCtx.lineTo(w * 0.85, h * 0.55);
  retCtx.stroke();
  retCtx.restore();

  retCtx.fillStyle = "#a855f7";
  retCtx.font = "bold 11px JetBrains Mono, monospace";
  retCtx.fillText("▲ ENTRY (IN) / ▼ EXIT (OUT) TRIPWIRE", w * 0.20, h * 0.53);

  // Draw Checkout Queue Zone Box
  const qx = w * 0.65, qy = h * 0.20, qw = w * 0.30, qh = h * 0.50;
  retCtx.fillStyle = "rgba(245, 158, 11, 0.08)";
  retCtx.fillRect(qx, qy, qw, qh);
  retCtx.strokeStyle = "rgba(245, 158, 11, 0.5)";
  retCtx.strokeRect(qx, qy, qw, qh);
  retCtx.fillStyle = "#f59e0b";
  retCtx.fillText("QUEUE ZONE: 3 WAITING", qx + 8, qy + 18);

  // FPS
  retFrameCount++;
  const now = performance.now();
  if (now - retLastFpsTime >= 1000) {
    const fps = Math.round((retFrameCount * 1000) / (now - retLastFpsTime));
    const fpsElem = document.getElementById("retFps");
    if (fpsElem) fpsElem.textContent = `${fps} FPS`;
    retFrameCount = 0;
    retLastFpsTime = now;
  }

  requestAnimationFrame(renderRetailLoop);
}

function simulateFootfallEvent(direction) {
  if (direction === 'in') {
    countIn++;
  } else {
    countOut++;
  }
  const inElem = document.getElementById("valIn");
  const outElem = document.getElementById("valOut");
  const curElem = document.getElementById("valCurrent");

  if (inElem) inElem.textContent = countIn;
  if (outElem) outElem.textContent = countOut;
  if (curElem) curElem.textContent = Math.max(0, countIn - countOut);
}

document.addEventListener("DOMContentLoaded", () => {
  loadRetailClip();
});
