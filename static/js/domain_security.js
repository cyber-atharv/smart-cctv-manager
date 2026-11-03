// ==================== UNIVERSAL SMART VISION: SECURITY DOMAIN ENGINE ====================
// Client-side 60 FPS rendering, polygon tripwire checks, weapon simulation, explainable scoring

let secStream = null;
let secVideo = null;
let secCanvas = null;
let secCtx = null;
let secAnimationId = null;

let isIntrusionActive = false;
let isWeaponActive = false;
let isLoiteringActive = false;
let loiteringTimer = 0;

// Tripwire coordinate parameters (normalized 0 to 1)
const TRIPWIRE = { x1: 0.20, y1: 0.70, x2: 0.80, y2: 0.70 };

async function startSecurityCam() {
  secVideo = document.getElementById("secVideo");
  secCanvas = document.getElementById("secCanvas");
  if (!secCanvas) return;
  secCtx = secCanvas.getContext("2d");

  try {
    secStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60, min: 30 } },
      audio: false
    });
    secVideo.srcObject = secStream;
    await secVideo.play();
    requestAnimationFrame(renderSecurityLoop);
  } catch (err) {
    console.warn("Using simulated synthetic stream for security domain:", err);
    loadSecurityClip();
  }
}

function loadSecurityClip() {
  secCanvas = document.getElementById("secCanvas");
  if (!secCanvas) return;
  secCtx = secCanvas.getContext("2d");
  secVideo = document.getElementById("secVideo");
  secVideo.srcObject = null;
  secVideo.src = "/static/videos/cv_demo_01.mp4";
  secVideo.loop = true;
  secVideo.play().then(() => {
    requestAnimationFrame(renderSecurityLoop);
  }).catch(() => {
    requestAnimationFrame(renderSecurityLoop);
  });
}

let secFrameCount = 0;
let secLastFpsTime = performance.now();

function renderSecurityLoop() {
  if (!secCanvas || !secCtx) return;
  const w = secCanvas.width;
  const h = secCanvas.height;

  // Draw base video or synthetic night-vision stream
  if (secVideo && secVideo.readyState >= 2) {
    secCtx.drawImage(secVideo, 0, 0, w, h);
  } else {
    // Dark facility synthetic feed
    secCtx.fillStyle = "#0c0d12";
    secCtx.fillRect(0, 0, w, h);
    secCtx.strokeStyle = "#1e2230";
    secCtx.lineWidth = 1;
    for (let x = 0; x < w; x += 60) {
      secCtx.beginPath(); secCtx.moveTo(x, 0); secCtx.lineTo(x, h); secCtx.stroke();
    }
    for (let y = 0; y < h; y += 60) {
      secCtx.beginPath(); secCtx.moveTo(0, y); secCtx.lineTo(w, y); secCtx.stroke();
    }
  }

  // Draw Virtual Tripwire Line
  const tx1 = TRIPWIRE.x1 * w, ty1 = TRIPWIRE.y1 * h;
  const tx2 = TRIPWIRE.x2 * w, ty2 = TRIPWIRE.y2 * h;

  secCtx.save();
  secCtx.strokeStyle = isIntrusionActive ? "rgba(239, 68, 68, 0.9)" : "rgba(6, 182, 212, 0.8)";
  secCtx.lineWidth = 3;
  secCtx.setLineDash([8, 6]);
  secCtx.beginPath();
  secCtx.moveTo(tx1, ty1);
  secCtx.lineTo(tx2, ty2);
  secCtx.stroke();
  secCtx.restore();

  // Tripwire Label
  secCtx.fillStyle = isIntrusionActive ? "#ef4444" : "#06b6d4";
  secCtx.font = "bold 12px JetBrains Mono, monospace";
  secCtx.fillText(isIntrusionActive ? "⚠ TRIPWIRE BREACH: INTRUSION" : "--- VIRTUAL TRIPWIRE LINE ---", tx1 + 10, ty1 - 8);

  // Draw Person Tracking Box
  const px = isIntrusionActive ? w * 0.45 : w * 0.35;
  const py = isIntrusionActive ? h * 0.58 : h * 0.30;
  const pw = 90, ph = 180;

  secCtx.strokeStyle = isWeaponActive ? "#ef4444" : (isIntrusionActive ? "#f97316" : "#22c55e");
  secCtx.lineWidth = 2;
  secCtx.strokeRect(px, py, pw, ph);

  // Person Label
  secCtx.fillStyle = isWeaponActive ? "#ef4444" : (isIntrusionActive ? "#f97316" : "#22c55e");
  secCtx.font = "11px Inter, sans-serif";
  secCtx.fillText("TRACK #01 [PERSON]", px, py - 6);

  // Draw Weapon Tag if active
  if (isWeaponActive) {
    const wx = px + pw - 20, wy = py + 60, ww = 45, wh = 35;
    secCtx.strokeStyle = "#ef4444";
    secCtx.lineWidth = 2;
    secCtx.strokeRect(wx, wy, ww, wh);

    secCtx.fillStyle = "rgba(239, 68, 68, 0.85)";
    secCtx.fillRect(wx, wy - 16, ww + 40, 16);
    secCtx.fillStyle = "#ffffff";
    secCtx.font = "bold 9px JetBrains Mono, monospace";
    secCtx.fillText("WEAPON 94%", wx + 2, wy - 4);
  }

  // Update FPS Meter
  secFrameCount++;
  const now = performance.now();
  if (now - secLastFpsTime >= 1000) {
    const fps = Math.round((secFrameCount * 1000) / (now - secLastFpsTime));
    const fpsElem = document.getElementById("secFps");
    if (fpsElem) fpsElem.textContent = `${fps} FPS`;
    secFrameCount = 0;
    secLastFpsTime = now;
  }

  requestAnimationFrame(renderSecurityLoop);
}

function simulateSecurityThreat(type) {
  if (type === 'weapon') {
    isWeaponActive = true;
  } else if (type === 'intrusion') {
    isIntrusionActive = true;
  }
  updateExplainableScoreUI();
}

function resetSecurityState() {
  isIntrusionActive = false;
  isWeaponActive = false;
  isLoiteringActive = false;
  updateExplainableScoreUI();
}

function updateExplainableScoreUI() {
  let score = 5; // Base presence
  const factorIntrusion = document.getElementById("factorIntrusion");
  const factorWeapon = document.getElementById("factorWeapon");
  const factorLoitering = document.getElementById("factorLoitering");
  const scoreVal = document.getElementById("threatScoreVal");
  const badge = document.getElementById("threatBadge");
  const bar = document.getElementById("threatProgressBar");
  const dispatch = document.getElementById("secDispatchStatus");

  if (factorIntrusion) factorIntrusion.style.display = isIntrusionActive ? "flex" : "none";
  if (factorWeapon) factorWeapon.style.display = isWeaponActive ? "flex" : "none";
  if (factorLoitering) factorLoitering.style.display = isLoiteringActive ? "flex" : "none";

  if (isIntrusionActive) score += 35;
  if (isWeaponActive) score += 35;
  if (isLoiteringActive) score += 20;

  if (scoreVal) scoreVal.innerHTML = `${score}<span style="font-size: 18px; color: var(--text-dim);">/100</span>`;
  if (bar) bar.style.width = `${score}%`;

  if (score >= 70) {
    badge.textContent = "● CRITICAL THREAT DETECTED";
    badge.style.background = "rgba(239, 68, 68, 0.2)";
    badge.style.color = "var(--danger)";
    bar.style.background = "var(--danger)";
    if (dispatch) {
      dispatch.textContent = "🚨 AUTOMATED SECURITY LOCKDOWN DISPATCHED";
      dispatch.style.color = "var(--danger)";
    }
  } else if (score >= 35) {
    badge.textContent = "● ELEVATED INCIDENT RISK";
    badge.style.background = "rgba(249, 115, 22, 0.2)";
    badge.style.color = "#f97316";
    bar.style.background = "#f97316";
    if (dispatch) {
      dispatch.textContent = "⚠ SECURITY DESK NOTIFIED";
      dispatch.style.color = "#f97316";
    }
  } else {
    badge.textContent = "● NORMAL SURVEILLANCE";
    badge.style.background = "rgba(34, 197, 94, 0.15)";
    badge.style.color = "var(--success)";
    bar.style.background = "var(--success)";
    if (dispatch) {
      dispatch.textContent = "STANDBY (ALL GREEN)";
      dispatch.style.color = "var(--success)";
    }
  }
}

function stopSecurityCam() {
  if (secStream && typeof secStream.getTracks === "function") {
    secStream.getTracks().forEach(t => { try { t.stop(); } catch(e){} });
    secStream = null;
  }
}
window.addEventListener("beforeunload", stopSecurityCam);
window.addEventListener("pagehide", stopSecurityCam);

document.addEventListener("DOMContentLoaded", () => {
  // Wait for user interaction or load clip by default
  loadSecurityClip();
});

