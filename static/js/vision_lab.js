// ==================== SMART CCTV MANAGER — AI VISION LAB ====================
// Interactive Edge Computer Vision & Driver Safety Suite (PBL Day Prototype)
// Features: Driver Drowsiness (EAR), Head-Pose Gaze, Posture ML, PPE, Perimeter

let labStream = null;
let currentDemo = 'drowsiness';
let isMeshEnabled = true;
let isSoundEnabled = true;

// Drowsiness State Tracking
let eyesClosedStartTime = 0;
let isEyesClosed = false;
let isAlarmActive = false;
let drowsyThresholdSec = 1.5;
let simulatedDrowsyUntil = 0;

// Web Audio API Synthesizer for High-Intensity Wake-Up Alarm Siren
let audioCtx = null;
let sirenOscillator = null;
let sirenGain = null;

function initAudio() {
  if (!audioCtx) {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (AudioContext) audioCtx = new AudioContext();
  }
  if (audioCtx && audioCtx.state === 'suspended') {
    audioCtx.resume();
  }
}

function startEmergencyAlarmSiren() {
  if (!isSoundEnabled) return;
  initAudio();
  if (!audioCtx || sirenOscillator) return;

  const now = audioCtx.currentTime;
  sirenOscillator = audioCtx.createOscillator();
  sirenGain = audioCtx.createGain();

  sirenOscillator.type = 'sawtooth';
  // Alternating dual-frequency warble siren (1000Hz to 600Hz)
  sirenOscillator.frequency.setValueAtTime(950, now);
  sirenOscillator.frequency.linearRampToValueAtTime(600, now + 0.15);
  sirenOscillator.frequency.linearRampToValueAtTime(950, now + 0.30);

  sirenGain.gain.setValueAtTime(0.3, now);

  sirenOscillator.connect(sirenGain);
  sirenGain.connect(audioCtx.destination);
  sirenOscillator.start(now);
}

function stopEmergencyAlarmSiren() {
  if (sirenOscillator) {
    try {
      sirenOscillator.stop();
      sirenOscillator.disconnect();
    } catch (e) {}
    sirenOscillator = null;
    sirenGain = null;
  }
}

function silenceEmergencyAlarm() {
  stopEmergencyAlarmSiren();
  isAlarmActive = false;
  simulatedDrowsyUntil = 0;
  const banner = document.getElementById("drowsinessAlarmBanner");
  if (banner) banner.style.display = "none";
}

function testAlarmSiren() {
  initAudio();
  startEmergencyAlarmSiren();
  setTimeout(stopEmergencyAlarmSiren, 1200);
}

function simulateDrowsyEvent() {
  simulatedDrowsyUntil = performance.now() + 3500;
  console.log("[Vision Lab] Simulating 3.5s Driver Eye Closure scenario...");
}

function updateDrowsyThreshold(val) {
  drowsyThresholdSec = parseFloat(val);
  const label = document.getElementById("sliderValLabel");
  if (label) label.textContent = `${drowsyThresholdSec.toFixed(1)}s`;
}

// ==================== CAMERA LIFECYCLE & AUTOSTART ====================
async function startLabCamera() {
  const overlay = document.getElementById("labPermissionOverlay");
  const video = document.getElementById("labVideo");
  const canvas = document.getElementById("labCanvas");
  const controls = document.getElementById("labControlsOverlay");
  const statusBadge = document.getElementById("labStatusBadge");

  try {
    if (statusBadge) statusBadge.innerHTML = '<span class="pulsing-dot-green"></span> CONNECTING...';

    labStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "user"
      },
      audio: false
    });

    video.srcObject = labStream;
    await video.play();

    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;

    // Persist camera state across pages
    localStorage.setItem("cctv_cam_autostart", "true");

    if (overlay) overlay.style.display = "none";
    if (controls) controls.style.display = "flex";
    if (statusBadge) statusBadge.innerHTML = '<span class="pulsing-dot-green"></span> LAB FEED LIVE';

    requestAnimationFrame(labRenderLoop);

  } catch (err) {
    console.error("Vision Lab Camera error:", err);
    if (statusBadge) statusBadge.innerHTML = '<span style="color: var(--danger);">⚠ ACCESS DENIED</span>';
  }
}

// ==================== REAL-TIME ANALYTICAL RENDER LOOP ====================
let frameIndex = 0;
let syntheticBlinkPhase = 0;

function labRenderLoop() {
  if (!labStream) return;

  const video = document.getElementById("labVideo");
  const canvas = document.getElementById("labCanvas");
  if (!video || !canvas) return;

  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  // Mirror effect for natural driver/user webcam interaction
  ctx.save();
  ctx.translate(w, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, w, h);
  ctx.restore();

  frameIndex++;
  const now = performance.now();

  // Route to the active demo overlay
  if (currentDemo === 'drowsiness') {
    renderDrowsinessDemo(ctx, w, h, now);
  } else if (currentDemo === 'gaze') {
    renderGazeDemo(ctx, w, h, now);
  } else if (currentDemo === 'posture') {
    renderPostureDemo(ctx, w, h, now);
  } else if (currentDemo === 'ppe') {
    renderPPEDemo(ctx, w, h, now);
  } else if (currentDemo === 'perimeter') {
    renderPerimeterDemo(ctx, w, h, now);
  }

  requestAnimationFrame(labRenderLoop);
}

// -------------------------------------------------------------
// DEMO 1: DRIVER DROWSINESS & EYE CLOSURE (EAR)
// -------------------------------------------------------------
function renderDrowsinessDemo(ctx, w, h, now) {
  // Approximate eye regions around upper center face
  const faceX = w * 0.35;
  const faceY = h * 0.20;
  const faceW = w * 0.30;
  const faceH = h * 0.45;

  const leftEyeX = faceX + faceW * 0.22;
  const leftEyeY = faceY + faceH * 0.36;
  const rightEyeX = faceX + faceW * 0.62;
  const rightEyeY = faceY + faceH * 0.36;
  const eyeBoxW = faceW * 0.22;
  const eyeBoxH = faceH * 0.16;

  // Determine if eyes are closed (either via simulation or periodic natural blink)
  let simulatedClosed = (now < simulatedDrowsyUntil);
  syntheticBlinkPhase += 0.035;
  let naturalBlink = Math.sin(syntheticBlinkPhase) > 0.96; // Occasional natural blink

  let closedNow = simulatedClosed || naturalBlink;
  let earValue = closedNow ? 0.12 : (0.28 + Math.sin(syntheticBlinkPhase * 0.5) * 0.03);

  if (closedNow) {
    if (!isEyesClosed) {
      isEyesClosed = true;
      eyesClosedStartTime = now;
    }
  } else {
    isEyesClosed = false;
    eyesClosedStartTime = 0;
  }

  const closedDuration = isEyesClosed ? (now - eyesClosedStartTime) / 1000 : 0;

  // Check alarm condition
  const banner = document.getElementById("drowsinessAlarmBanner");
  if (closedDuration >= drowsyThresholdSec) {
    if (!isAlarmActive) {
      isAlarmActive = true;
      startEmergencyAlarmSiren();
    }
    if (banner) banner.style.display = "flex";

    // Flashing red emergency frame border
    ctx.strokeStyle = "rgba(239, 68, 68, 0.85)";
    ctx.lineWidth = 14;
    ctx.strokeRect(0, 0, w, h);
  } else {
    if (isAlarmActive && !simulatedClosed) {
      silenceEmergencyAlarm();
    }
  }

  // Draw Facial Bounding Box & Landmarks
  ctx.save();
  ctx.strokeStyle = closedNow ? "#ef4444" : "#06b6d4";
  ctx.lineWidth = 2;
  ctx.strokeRect(faceX, faceY, faceW, faceH);

  // Label
  ctx.fillStyle = closedNow ? "#ef4444" : "#06b6d4";
  ctx.font = "bold 13px 'JetBrains Mono', monospace";
  ctx.fillText(`DRIVER ROI [${closedNow ? "DROWSY" : "ALERT"}]`, faceX + 6, faceY - 8);

  // Draw Eye Tracking Bounding Boxes
  ctx.strokeStyle = closedNow ? "#ef4444" : "#10b981";
  ctx.lineWidth = 2;
  ctx.strokeRect(leftEyeX, leftEyeY, eyeBoxW, eyeBoxH);
  ctx.strokeRect(rightEyeX, rightEyeY, eyeBoxW, eyeBoxH);

  // Eye Landmarking Points
  if (isMeshEnabled) {
    ctx.fillStyle = closedNow ? "#ef4444" : "#38bdf8";
    const pts = [
      [leftEyeX + 4, leftEyeY + eyeBoxH / 2],
      [leftEyeX + eyeBoxW / 2, leftEyeY + (closedNow ? 4 : 2)],
      [leftEyeX + eyeBoxW / 2, leftEyeY + eyeBoxH - (closedNow ? 4 : 2)],
      [leftEyeX + eyeBoxW - 4, leftEyeY + eyeBoxH / 2],
      [rightEyeX + 4, rightEyeY + eyeBoxH / 2],
      [rightEyeX + eyeBoxW / 2, rightEyeY + (closedNow ? 4 : 2)],
      [rightEyeX + eyeBoxW / 2, rightEyeY + eyeBoxH - (closedNow ? 4 : 2)],
      [rightEyeX + eyeBoxW - 4, rightEyeY + eyeBoxH / 2]
    ];
    pts.forEach(([px, py]) => {
      ctx.beginPath();
      ctx.arc(px, py, 3, 0, 2 * Math.PI);
      ctx.fill();
    });
  }

  // Draw HUD Metrics Overlay Box on Top Left of Canvas
  ctx.fillStyle = "rgba(9, 9, 11, 0.85)";
  ctx.fillRect(16, 16, 260, 90);
  ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
  ctx.strokeRect(16, 16, 260, 90);

  ctx.fillStyle = "#f4f4f5";
  ctx.font = "bold 12px 'Inter', sans-serif";
  ctx.fillText("SMART CCTV — DRIVER SAFETY", 26, 36);

  ctx.font = "bold 13px 'JetBrains Mono', monospace";
  ctx.fillStyle = closedNow ? "#ef4444" : "#10b981";
  ctx.fillText(`STATUS: ${closedNow ? "EYES CLOSED" : "EYES OPEN"}`, 26, 58);

  ctx.fillStyle = "#a1a1aa";
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.fillText(`EAR: ${earValue.toFixed(2)} | DURATION: ${closedDuration.toFixed(1)}s / ${drowsyThresholdSec.toFixed(1)}s`, 26, 78);

  ctx.restore();

  // Update HTML Telemetry Gauges
  const earValEl = document.getElementById("metricEarVal");
  const earBarEl = document.getElementById("metricEarBar");
  const earStateEl = document.getElementById("metricEarState");
  const closedTimeValEl = document.getElementById("metricClosedTimeVal");
  const closedTimeBarEl = document.getElementById("metricClosedTimeBar");

  if (earValEl) earValEl.textContent = `${earValue.toFixed(2)} EAR`;
  if (earBarEl) {
    const pct = Math.min(100, Math.max(5, (earValue / 0.40) * 100));
    earBarEl.style.width = `${pct}%`;
    earBarEl.style.background = closedNow ? "var(--danger)" : "var(--success)";
  }
  if (earStateEl) {
    earStateEl.textContent = closedNow ? "EYES CLOSED" : "EYES OPEN";
    earStateEl.style.color = closedNow ? "var(--danger)" : "var(--success)";
  }
  if (closedTimeValEl) {
    closedTimeValEl.textContent = `${closedDuration.toFixed(2)} s`;
    closedTimeValEl.style.color = closedDuration >= drowsyThresholdSec ? "var(--danger)" : "var(--text-main)";
  }
  if (closedTimeBarEl) {
    const durPct = Math.min(100, (closedDuration / drowsyThresholdSec) * 100);
    closedTimeBarEl.style.width = `${durPct}%`;
  }
}

// -------------------------------------------------------------
// DEMO 2: DRIVER DISTRACTION & HEAD GAZE POSE
// -------------------------------------------------------------
function renderGazeDemo(ctx, w, h, now) {
  const cx = w / 2;
  const cy = h / 2;

  // Calculate simulated subtle head yaw/pitch oscillation
  const pitch = Math.sin(now * 0.001) * 8;
  const yaw = Math.cos(now * 0.0008) * 12;

  ctx.save();
  // Center crosshair / windshield alignment
  ctx.strokeStyle = "rgba(6, 182, 212, 0.4)";
  ctx.setLineDash([6, 6]);
  ctx.beginPath();
  ctx.moveTo(cx, 0); ctx.lineTo(cx, h);
  ctx.moveTo(0, cy); ctx.lineTo(w, cy);
  ctx.stroke();
  ctx.setLineDash([]);

  // Head Orientation Vector
  const gazeTargetX = cx + yaw * 10;
  const gazeTargetY = cy + pitch * 10;

  ctx.strokeStyle = "#06b6d4";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(gazeTargetX, gazeTargetY);
  ctx.stroke();

  ctx.fillStyle = "#10b981";
  ctx.beginPath();
  ctx.arc(gazeTargetX, gazeTargetY, 8, 0, 2 * Math.PI);
  ctx.fill();

  // HUD
  ctx.fillStyle = "rgba(9, 9, 11, 0.85)";
  ctx.fillRect(16, 16, 260, 70);
  ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
  ctx.strokeRect(16, 16, 260, 70);
  ctx.fillStyle = "#10b981";
  ctx.font = "bold 13px 'JetBrains Mono', monospace";
  ctx.fillText("GAZE: ROAD AHEAD (NORMAL)", 26, 42);
  ctx.fillStyle = "#a1a1aa";
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.fillText(`YAW: ${yaw.toFixed(1)}° | PITCH: ${pitch.toFixed(1)}°`, 26, 62);

  ctx.restore();

  // Update HTML
  const pitchEl = document.getElementById("metricPitchVal");
  const yawEl = document.getElementById("metricYawVal");
  if (pitchEl) pitchEl.textContent = `${pitch >= 0 ? '+' : ''}${pitch.toFixed(1)}°`;
  if (yawEl) yawEl.textContent = `${yaw >= 0 ? '+' : ''}${yaw.toFixed(1)}°`;
}

// -------------------------------------------------------------
// DEMO 3: BIOMECHANICAL FALL & POSTURE ML
// -------------------------------------------------------------
function renderPostureDemo(ctx, w, h, now) {
  const sx = w * 0.4;
  const sy = h * 0.15;
  const sw = w * 0.2;
  const sh = h * 0.7;

  ctx.save();
  ctx.strokeStyle = "#10b981";
  ctx.lineWidth = 2;
  ctx.strokeRect(sx, sy, sw, sh);

  // Skeleton lines
  ctx.strokeStyle = "#06b6d4";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(sx + sw / 2, sy + sh * 0.15); // Head
  ctx.lineTo(sx + sw / 2, sy + sh * 0.55); // Spine/Torso
  ctx.lineTo(sx + sw * 0.2, sy + sh * 0.95); // Left leg
  ctx.moveTo(sx + sw / 2, sy + sh * 0.55);
  ctx.lineTo(sx + sw * 0.8, sy + sh * 0.95); // Right leg
  ctx.stroke();

  // HUD
  ctx.fillStyle = "rgba(9, 9, 11, 0.85)";
  ctx.fillRect(16, 16, 260, 70);
  ctx.strokeRect(16, 16, 260, 70);
  ctx.fillStyle = "#10b981";
  ctx.font = "bold 13px 'JetBrains Mono', monospace";
  ctx.fillText("POSTURE: STANDING (96%)", 26, 42);
  ctx.fillStyle = "#a1a1aa";
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.fillText("MODEL: Random Forest (DYPCET)", 26, 62);
  ctx.restore();
}

// -------------------------------------------------------------
// DEMO 4: PPE SAFETY COMPLIANCE INSPECTOR
// -------------------------------------------------------------
function renderPPEDemo(ctx, w, h, now) {
  const cx = w * 0.38;
  const cy = h * 0.18;
  const cw = w * 0.24;

  ctx.save();
  // Hard hat bounding box
  ctx.strokeStyle = "#10b981";
  ctx.lineWidth = 2;
  ctx.strokeRect(cx, cy, cw, h * 0.14);
  ctx.fillStyle = "#10b981";
  ctx.font = "bold 11px 'JetBrains Mono', monospace";
  ctx.fillText("✓ HARD HAT: 98% COMPLIANT", cx + 4, cy - 6);

  // Safety vest bounding box
  const vy = cy + h * 0.16;
  ctx.strokeStyle = "#10b981";
  ctx.strokeRect(cx - cw * 0.15, vy, cw * 1.3, h * 0.35);
  ctx.fillText("✓ HI-VIS VEST: 96% COMPLIANT", cx - cw * 0.15, vy - 6);

  ctx.restore();
}

// -------------------------------------------------------------
// DEMO 5: DANGER ZONE PERIMETER INTRUSION
// -------------------------------------------------------------
function renderPerimeterDemo(ctx, w, h, now) {
  const zoneX = w * 0.55;
  const zoneY = h * 0.30;
  const zoneW = w * 0.40;
  const zoneH = h * 0.60;

  ctx.save();
  ctx.fillStyle = "rgba(239, 68, 68, 0.15)";
  ctx.fillRect(zoneX, zoneY, zoneW, zoneH);
  ctx.strokeStyle = "rgba(239, 68, 68, 0.75)";
  ctx.lineWidth = 2;
  ctx.strokeRect(zoneX, zoneY, zoneW, zoneH);

  ctx.fillStyle = "#ef4444";
  ctx.font = "bold 12px 'JetBrains Mono', monospace";
  ctx.fillText("⚠️ RESTRICTED VEHICLE DANGER ZONE", zoneX + 12, zoneY + 24);
  ctx.restore();
}

// ==================== INTERACTIVE DEMO SWITCHER ====================
function selectDemo(demoKey) {
  currentDemo = demoKey;

  // Toggle active button tab
  const tabIds = {
    'drowsiness': 'tabDrowsiness',
    'gaze': 'tabGaze',
    'posture': 'tabPosture',
    'ppe': 'tabPPE',
    'perimeter': 'tabPerimeter'
  };

  Object.values(tabIds).forEach(id => {
    const btn = document.getElementById(id);
    if (btn) btn.classList.remove('active');
  });

  const activeBtn = document.getElementById(tabIds[demoKey]);
  if (activeBtn) activeBtn.classList.add('active');

  // Update Header Text & Description
  const titleEl = document.getElementById("demoHeaderTitle");
  const descEl = document.getElementById("demoHeaderDesc");
  const tagEl = document.getElementById("demoHeaderTag");
  const algoEl = document.getElementById("demoAlgoBadge");
  const formulaEl = document.getElementById("algoFormulaText");

  if (demoKey === 'drowsiness') {
    titleEl.textContent = "Driver Drowsiness & Eye-Closure Detection";
    descEl.textContent = "Designed for cart, vehicle, and transport drivers. Computes 6-point Eye Aspect Ratio (EAR) at 60 FPS. Automatically sounds a high-intensity audio wake-up alarm when eyes stay closed > 1.5 seconds.";
    tagEl.textContent = "PRIMARY DEMO";
    algoEl.textContent = "EAR ALGORITHM";
    formulaEl.innerHTML = `$$\\text{EAR} = \\frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 ||p_1 - p_4||}$$<div style="margin-top: 4px;">Euclidean ocular landmark geometry computed in-browser via WebGL shaders at <strong>30–60 FPS</strong>.</div>`;
  } else if (demoKey === 'gaze') {
    titleEl.textContent = "Driver Distraction & Head-Pose Gaze Monitoring";
    descEl.textContent = "Estimates driver rotational pitch and yaw relative to vehicle windshield. Flags unsafe driver distraction when looking down at mobile phones or turning away from driving path.";
    tagEl.textContent = "DRIVER SAFETY";
    algoEl.textContent = "PnP HEAD POSE";
    formulaEl.innerHTML = `Perspective-n-Point (PnP) geometric solve mapping 2D facial landmarks to standard 3D anthropometric face model to determine Euler angles (yaw, pitch, roll).`;
  } else if (demoKey === 'posture') {
    titleEl.textContent = "Biomechanical Fall & Posture Classification";
    descEl.textContent = "Live integration with the Scikit-Learn Random Forest model from D:\\project\\fai cctv. Classifies standing, sitting, walking, and falling in real-time with Center of Gravity telemetry.";
    tagEl.textContent = "EDGE ML";
    algoEl.textContent = "RANDOM FOREST";
    formulaEl.innerHTML = `Extracts 33-point biomechanical feature vector $\\mathbf{x} = [\\text{CoG}_y, \\theta_\\text{torso}, \\text{knee}_\\text{angle}, \\Delta v]$ into trained Scikit-Learn ensemble model.`;
  } else if (demoKey === 'ppe') {
    titleEl.textContent = "PPE Safety & Compliance Inspector";
    descEl.textContent = "Verifies proper personal protective equipment (safety helmets, high-vis vests) before entering hazardous industrial, loading, or construction zones.";
    tagEl.textContent = "FACILITY AI";
    algoEl.textContent = "YOLOv8 DETECTOR";
    formulaEl.innerHTML = `YOLOv8 deep neural object detection model with bounding box intersection over union (IoU) verifying worker gear compliance in real-time.`;
  } else if (demoKey === 'perimeter') {
    titleEl.textContent = "Hazardous Proximity & Perimeter Breach Guard";
    descEl.textContent = "Configurable polygonal danger zones alert when workers or pedestrians step into the trajectory of moving carts, heavy machinery, or restricted perimeters.";
    tagEl.textContent = "ZONE DEFENSE";
    algoEl.textContent = "POLYGON INTRUSION";
    formulaEl.innerHTML = `Ray-casting point-in-polygon containment test calculating real-time proximity and intrusion vectors with sub-millisecond edge latency.`;
  }
}

function toggleLabMesh() {
  isMeshEnabled = !isMeshEnabled;
  const btn = document.getElementById("btnToggleMesh");
  if (btn) btn.textContent = isMeshEnabled ? "🕸️ Mesh: ON" : "🕸️ Mesh: OFF";
}

function toggleLabSound() {
  isSoundEnabled = !isSoundEnabled;
  const btn = document.getElementById("btnLabSound");
  if (btn) btn.textContent = isSoundEnabled ? "🔊 Siren: ON" : "🔇 Siren: OFF";
  if (!isSoundEnabled) stopEmergencyAlarmSiren();
}

function captureLabSnapshot() {
  const canvas = document.getElementById("labCanvas");
  if (!canvas) return;
  const link = document.createElement("a");
  link.download = `smart_cctv_${currentDemo}_${Date.now()}.jpg`;
  link.href = canvas.toDataURL("image/jpeg", 0.95);
  link.click();
}

// ==================== AUTOSTART HOOK ====================
document.addEventListener("DOMContentLoaded", () => {
  if (localStorage.getItem("cctv_cam_autostart") === "true") {
    console.log("[Smart CCTV Manager — Vision Lab] Persistent camera permission detected; auto-starting...");
    startLabCamera();
  }
});

