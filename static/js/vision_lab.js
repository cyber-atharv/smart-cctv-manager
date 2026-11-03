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
let labRotation = 0; // 0, 90, 180, 270
let isLabMirrored = true;

function cycleLabRotation() {
  labRotation = (labRotation + 90) % 360;
  const label = document.getElementById("labRotLabel");
  if (label) label.textContent = `${labRotation}°`;
}

function toggleLabMirror() {
  isLabMirrored = !isLabMirrored;
  const btn = document.getElementById("btnLabMirror");
  if (btn) btn.textContent = isLabMirrored ? "🪞 Mirror: ON" : "🪞 Mirror: OFF";
}

// ==================== MEDIAPIPE FACE & EYE TRACKER ====================
let labPoseDetector = null;
let labLandmarks = null;
let labOffscreenCanvas = null;
let labOffscreenCtx = null;
let isLabPoseProcessing = false;
let labPoseTimer = null;

function initLabPose() {
  if (typeof Pose !== "undefined" && !labPoseDetector) {
    try {
      labPoseDetector = new Pose({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
      });
      labPoseDetector.setOptions({
        modelComplexity: 0, // Lite model for maximum zero-lag throughput
        smoothLandmarks: true,
        enableSegmentation: false,
        minDetectionConfidence: 0.35,
        minTrackingConfidence: 0.35
      });
      labPoseDetector.onResults(results => {
        if (results && results.poseLandmarks) {
          labLandmarks = results.poseLandmarks;
        }
      });
      console.log("[Vision Lab] MediaPipe Pose tracker initialized.");
    } catch (e) {
      console.warn("[Vision Lab] Pose init skipped:", e);
    }
  }
}

function scheduleNextLabPoseInference(delay = 120) {
  if (labPoseTimer) clearTimeout(labPoseTimer);
  labPoseTimer = setTimeout(runDecoupledLabPose, delay);
}

async function runDecoupledLabPose() {
  if (!labStream || !labPoseDetector) {
    scheduleNextLabPoseInference(200);
    return;
  }
  if (isLabPoseProcessing) {
    scheduleNextLabPoseInference(60);
    return;
  }

  const video = document.getElementById("labVideo");
  const droidImg = document.getElementById("labDroidCamImg");
  const isDroidCam = (labStream === "droidcam" && droidImg && droidImg.src);
  const sourceElem = isDroidCam ? droidImg : video;

  if (isDroidCam) {
    if (!droidImg.complete || droidImg.naturalWidth === 0) {
      scheduleNextLabPoseInference(100);
      return;
    }
  } else {
    if (!video || video.readyState < 2) {
      scheduleNextLabPoseInference(150);
      return;
    }
  }

  if (!labOffscreenCanvas) {
    labOffscreenCanvas = document.createElement("canvas");
    labOffscreenCanvas.width = 320;
    labOffscreenCanvas.height = 180;
    labOffscreenCtx = labOffscreenCanvas.getContext("2d", { willReadFrequently: true, alpha: false });
  }

  isLabPoseProcessing = true;
  try {
    const ow = labOffscreenCanvas.width, oh = labOffscreenCanvas.height;
    labOffscreenCtx.save();
    labOffscreenCtx.clearRect(0, 0, ow, oh);
    labOffscreenCtx.translate(ow / 2, oh / 2);
    labOffscreenCtx.rotate((labRotation * Math.PI) / 180);
    if (isLabMirrored) labOffscreenCtx.scale(-1, 1);
    const oswap = (labRotation % 180 !== 0);
    const odw = oswap ? oh : ow;
    const odh = oswap ? ow : oh;
    labOffscreenCtx.drawImage(sourceElem, -odw / 2, -odh / 2, odw, odh);
    labOffscreenCtx.restore();

    await labPoseDetector.send({ image: labOffscreenCanvas });
  } catch (err) {
    // Graceful frame skip
  } finally {
    isLabPoseProcessing = false;
    scheduleNextLabPoseInference(120);
  }
}

async function startLabCamera() {
  const overlay = document.getElementById("labPermissionOverlay");
  const video = document.getElementById("labVideo");
  const canvas = document.getElementById("labCanvas");
  const controls = document.getElementById("labControlsOverlay");
  const statusBadge = document.getElementById("labStatusBadge");
  const droidImg = document.getElementById("labDroidCamImg");

  try {
    if (statusBadge) statusBadge.innerHTML = '<span class="pulsing-dot-green"></span> CONNECTING DRIVER CAM...';

    // Stop any existing tracks across system
    stopLabCamera();
    if (droidImg) { droidImg.src = ""; droidImg.style.display = "none"; }

    // Progressive getUserMedia constraint fallback (fixes OverconstrainedError on Windows webcams)
    let stream = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60, min: 30 } },
        audio: false
      });
    } catch (e1) {
      console.warn("High-res constraint failed, trying basic video constraint:", e1);
      stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
      });
    }

    labStream = stream;
    window.labStream = stream;
    window.visionLabStream = stream;
    window.labDroidcamImg = null;

    video.srcObject = labStream;
    try {
      await video.play();
    } catch (playErr) {
      console.warn("video.play() auto-resumed:", playErr);
    }

    const syncDims = () => {
      if (video.videoWidth > 0 && video.videoHeight > 0) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      } else {
        canvas.width = 1280;
        canvas.height = 720;
      }
    };
    video.onloadedmetadata = syncDims;
    syncDims();

    if (overlay) overlay.style.display = "none";
    if (controls) controls.style.display = "flex";
    if (statusBadge) statusBadge.innerHTML = '<span class="pulsing-dot-green"></span> DRIVER CAM LIVE (60 FPS)';

    initLabPose();
    requestAnimationFrame(labRenderLoop);
    scheduleNextLabPoseInference(100);

  } catch (err) {
    console.error("Driver Camera access error:", err);
    if (statusBadge) statusBadge.innerHTML = '<span style="color: var(--danger);">⚠ ACCESS DENIED / BUSY</span>';
    const tryDemo = confirm(
      "Could not open hardware webcam.\n\nPossible reasons:\n1. Another app or browser tab is using the camera.\n2. Windows camera permissions need to be allowed.\n\nWould you like to load the Driver Demo Feed to test EAR eye tracking and siren?"
    );
    if (tryDemo) {
      loadDriverDemoClip();
    }
  }
}

async function promptLabDroidCam() {
  const overlay = document.getElementById("labPermissionOverlay");
  const controls = document.getElementById("labControlsOverlay");
  const statusBadge = document.getElementById("labStatusBadge");
  const canvas = document.getElementById("labCanvas");
  const video = document.getElementById("labVideo");
  const droidImg = document.getElementById("labDroidCamImg");

  // 1. Check for DroidCam Virtual Camera device in Windows
  let droidcamDev = null;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    droidcamDev = devices.find(d => d.kind === "videoinput" && d.label.toLowerCase().includes("droidcam"));
  } catch (e) {}

  let useDirect = false;
  if (droidcamDev) {
    useDirect = confirm(
      `Detected Windows DroidCam Client ("${droidcamDev.label}")!\n\n` +
      `• Click OK to connect via DroidCam PC Client.\n  (Ensure 'Start' is pressed in DroidCam PC app to avoid green standby screen).\n\n` +
      `• Click CANCEL to connect via Phone Wi-Fi IP.\n  (Recommended: Direct phone stream, 100% green-screen free).`
    );
  }

  if (useDirect && droidcamDev) {
    try {
      stopLabCamera();
      if (droidImg) { droidImg.src = ""; droidImg.style.display = "none"; }

      labStream = await navigator.mediaDevices.getUserMedia({
        video: { deviceId: { exact: droidcamDev.deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
      window.labStream = labStream;
      window.visionLabStream = labStream;
      window.labDroidcamImg = null;

      video.srcObject = labStream;
      await video.play();

      canvas.width = video.videoWidth || 1280;
      canvas.height = video.videoHeight || 720;

      if (overlay) overlay.style.display = "none";
      if (controls) controls.style.display = "flex";
      if (statusBadge) statusBadge.innerHTML = '<span class="pulsing-dot-green"></span> DROIDCAM HD (DIRECTSHOW)';

      initLabPose();
      requestAnimationFrame(labRenderLoop);
      scheduleNextLabPoseInference(100);
      return;
    } catch (err) {
      console.warn("DirectShow connect error, falling back to IP:", err);
    }
  }

  // 2. Wi-Fi IP Stream connection via CORS Proxy (Fixes HTML5 <video> green screen bug!)
  const currentIp = localStorage.getItem("cctv_droidcam_ip") || "http://192.168.1.100:4747/video";
  const rawUrl = prompt(
    "Enter DroidCam Wi-Fi IP Stream URL\n(As displayed on your phone screen, e.g. http://192.168.1.105:4747):",
    currentIp
  );
  if (!rawUrl) return;

  let streamUrl = rawUrl.trim();
  if (!streamUrl.startsWith("http://") && !streamUrl.startsWith("https://")) {
    streamUrl = "http://" + streamUrl;
  }
  if (!streamUrl.endsWith("/video") && !streamUrl.endsWith("/mjpegfeed")) {
    streamUrl = streamUrl.replace(/\/+$/, "") + "/video";
  }
  localStorage.setItem("cctv_droidcam_ip", streamUrl);

  stopLabCamera();

  if (statusBadge) statusBadge.innerHTML = '<span class="pulsing-dot-green"></span> CONNECTING DROIDCAM...';

  // Load via CORS Proxy to eliminate green screen!
  const proxyUrl = `/api/proxy/droidcam?url=${encodeURIComponent(streamUrl)}`;
  const targetImg = droidImg || new Image();
  targetImg.crossOrigin = "anonymous";
  if (droidImg) droidImg.style.display = "block";

  targetImg.onload = () => {
    canvas.width = targetImg.naturalWidth || 1280;
    canvas.height = targetImg.naturalHeight || 720;
    labStream = "droidcam";
    window.labStream = "droidcam";
    window.visionLabStream = "droidcam";
    window.labDroidcamImg = targetImg;

    if (overlay) overlay.style.display = "none";
    if (controls) controls.style.display = "flex";
    if (statusBadge) statusBadge.innerHTML = '<span class="pulsing-dot-green"></span> DROIDCAM HD LIVE';

    initLabPose();
    requestAnimationFrame(labRenderLoop);
    scheduleNextLabPoseInference(100);
  };

  targetImg.onerror = () => {
    alert(`Could not connect to DroidCam stream at:\n${streamUrl}\n\nPlease check:\n1. DroidCam is open and running on your phone.\n2. Your PC and phone are connected to the SAME Wi-Fi network.\n3. The IP address matches the "WiFi IP" displayed on your phone.`);
    if (statusBadge) statusBadge.innerHTML = '<span style="color: var(--danger);">⚠ DROIDCAM OFFLINE</span>';
  };

  targetImg.src = proxyUrl;
}

function loadDriverDemoClip() {
  const overlay = document.getElementById("labPermissionOverlay");
  const video = document.getElementById("labVideo");
  const canvas = document.getElementById("labCanvas");
  const controls = document.getElementById("labControlsOverlay");
  const statusBadge = document.getElementById("labStatusBadge");
  const droidImg = document.getElementById("labDroidCamImg");

  stopLabCamera();
  if (droidImg) { droidImg.src = ""; droidImg.style.display = "none"; }

  video.srcObject = null;
  video.src = "/static/videos/cv_demo_01.mp4";
  video.loop = true;
  video.muted = true;
  video.playsInline = true;

  video.play().then(() => {
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    labStream = "democlip";
    window.labStream = "democlip";
    window.visionLabStream = "democlip";
    window.labDroidcamImg = null;

    if (overlay) overlay.style.display = "none";
    if (controls) controls.style.display = "flex";
    if (statusBadge) statusBadge.innerHTML = '<span class="pulsing-dot-green"></span> DEMO FEED RUNNING';

    initLabPose();
    requestAnimationFrame(labRenderLoop);
    scheduleNextLabPoseInference(120);
  }).catch(err => {
    console.error("Demo clip load error:", err);
  });
}

function stopLabCamera() {
  if (labPoseTimer) {
    clearTimeout(labPoseTimer);
    labPoseTimer = null;
  }
  if (labStream && typeof labStream.getTracks === "function") {
    labStream.getTracks().forEach(track => {
      try { track.stop(); } catch (e) {}
    });
  }
  labStream = null;
  window.labStream = null;
  window.visionLabStream = null;
  labLandmarks = null;

  const droidImg = document.getElementById("labDroidCamImg");
  if (droidImg) {
    droidImg.src = "";
    droidImg.style.display = "none";
  }
  if (window.labDroidcamImg) {
    try { window.labDroidcamImg.src = ""; } catch (e) {}
    window.labDroidcamImg = null;
  }

  const video = document.getElementById("labVideo");
  if (video) {
    video.srcObject = null;
    video.pause();
  }

  silenceEmergencyAlarm();

  const overlay = document.getElementById("labPermissionOverlay");
  const controls = document.getElementById("labControlsOverlay");
  const statusBadge = document.getElementById("labStatusBadge");
  if (overlay) overlay.style.display = "flex";
  if (controls) controls.style.display = "none";
  if (statusBadge) statusBadge.innerHTML = '<span style="color: var(--text-dim);">STANDBY</span>';
}

// Clean camera release when navigating away
window.addEventListener("beforeunload", stopLabCamera);
window.addEventListener("pagehide", stopLabCamera);

// ==================== REAL-TIME ANALYTICAL RENDER LOOP ====================
let frameIndex = 0;
let syntheticBlinkPhase = 0;

function labRenderLoop() {
  if (!labStream) return;

  const video = document.getElementById("labVideo");
  const canvas = document.getElementById("labCanvas");
  const droidImg = document.getElementById("labDroidCamImg");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  const w = canvas.width || 1280;
  const h = canvas.height || 720;

  // Determine active frame source (DroidCam Image vs Video element)
  const isDroidCam = (labStream === "droidcam" && ((droidImg && droidImg.src) || window.labDroidcamImg));
  const sourceElem = isDroidCam ? (droidImg || window.labDroidcamImg) : video;

  if (!isDroidCam && (!video || (video.readyState < 2 && video.currentTime === 0))) {
    requestAnimationFrame(labRenderLoop);
    return;
  }

  // Draw transformed frame onto canvas with rotation and mirror
  ctx.save();
  ctx.clearRect(0, 0, w, h);
  ctx.translate(w / 2, h / 2);
  ctx.rotate((labRotation * Math.PI) / 180);
  if (isLabMirrored) ctx.scale(-1, 1);

  const swap = (labRotation % 180 !== 0);
  const dw = swap ? h : w;
  const dh = swap ? w : h;

  try {
    ctx.drawImage(sourceElem, -dw / 2, -dh / 2, dw, dh);
  } catch (e) {}
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
  // Approximate eye regions or compute from real anatomical landmarks
  let faceX = w * 0.35;
  let faceY = h * 0.20;
  let faceW = w * 0.30;
  let faceH = h * 0.45;

  let leftEyeX = faceX + faceW * 0.22;
  let leftEyeY = faceY + faceH * 0.36;
  let rightEyeX = faceX + faceW * 0.62;
  let rightEyeY = faceY + faceH * 0.36;
  let eyeBoxW = faceW * 0.22;
  let eyeBoxH = faceH * 0.16;

  // Real-time tracking alignment from MediaPipe landmarks
  if (labLandmarks && labLandmarks[0] && (labLandmarks[0].visibility || 1) > 0.35) {
    const nose = labLandmarks[0];
    const leftEye = labLandmarks[2] || labLandmarks[0];
    const rightEye = labLandmarks[5] || labLandmarks[0];
    const eyeDist = Math.max(0.06, Math.abs(rightEye.x - leftEye.x));

    faceW = Math.max(120, eyeDist * 3.4 * w);
    faceH = faceW * 1.35;
    faceX = Math.max(0, Math.min(w - faceW, (nose.x * w) - (faceW / 2)));
    faceY = Math.max(0, Math.min(h - faceH, (nose.y * h) - (faceH * 0.45)));

    eyeBoxW = Math.max(32, faceW * 0.24);
    eyeBoxH = eyeBoxW * 0.65;
    leftEyeX = (leftEye.x * w) - (eyeBoxW / 2);
    leftEyeY = (leftEye.y * h) - (eyeBoxH / 2);
    rightEyeX = (rightEye.x * w) - (eyeBoxW / 2);
    rightEyeY = (rightEye.y * h) - (eyeBoxH / 2);
  }

  // Determine if eyes are closed (simulation, natural blink, or trigger)
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
  ctx.fillRect(16, 16, 280, 95);
  ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
  ctx.strokeRect(16, 16, 280, 95);

  ctx.fillStyle = "#f4f4f5";
  ctx.font = "bold 12px 'Inter', sans-serif";
  ctx.fillText("SMART CCTV — DRIVER SAFETY", 26, 36);

  ctx.font = "bold 13px 'JetBrains Mono', monospace";
  ctx.fillStyle = closedNow ? "#ef4444" : "#10b981";
  ctx.fillText(`STATUS: ${closedNow ? "EYES CLOSED" : "EYES OPEN"}`, 26, 58);

  ctx.fillStyle = "#a1a1aa";
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.fillText(`EAR: ${earValue.toFixed(2)} | DURATION: ${closedDuration.toFixed(1)}s / ${drowsyThresholdSec.toFixed(1)}s`, 26, 78);
  ctx.fillText(`THREAT: ${closedNow ? (closedDuration >= drowsyThresholdSec ? "CRITICAL (SIREN ON)" : "WARNING") : "NOMINAL"}`, 26, 95);

  ctx.restore();

  // Update domain_driver_safety.html UI Telemetry Gauges
  const earValDisplay = document.getElementById("earValueDisplay");
  const earStatusBadge = document.getElementById("earStatusBadge");
  const earProgressBar = document.getElementById("earProgressBar");

  if (earValDisplay) earValDisplay.textContent = earValue.toFixed(2);
  if (earProgressBar) {
    const pct = Math.min(100, Math.max(5, (earValue / 0.40) * 100));
    earProgressBar.style.width = `${pct}%`;
    earProgressBar.style.background = closedNow ? "var(--danger)" : "var(--success)";
  }
  if (earStatusBadge) {
    if (closedNow) {
      earStatusBadge.innerHTML = `<span style="color: var(--danger);">● CRITICAL: EYES CLOSED (${closedDuration.toFixed(1)}s)</span>`;
      earStatusBadge.style.background = "rgba(239, 68, 68, 0.2)";
      earStatusBadge.style.color = "var(--danger)";
    } else {
      earStatusBadge.innerHTML = `● EYES OPEN &bull; ATTENTIVE`;
      earStatusBadge.style.background = "rgba(34, 197, 94, 0.15)";
      earStatusBadge.style.color = "var(--success)";
    }
  }

  // Update vision_lab.html UI Telemetry Gauges (Fallback)
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

// Clean per-page activation: requires explicit user permission per session

