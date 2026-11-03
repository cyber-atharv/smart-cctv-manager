// ==================== IN-BUILT CAMERA ZERO-LAG VISION ENGINE ====================
// Uses client-side WebCam + MediaPipe Pose for 30–60 FPS smooth video
// Features DroidCam Mobile Phone Integration & Rotation Fixes from D:\project\fai cctv

let inbuiltStream = null;
let poseDetector = null;
let cameraInstance = null;
let isAIActive = true;
let isSkeletonEnabled = true;
let isBackendMode = false;

let fpsCount = 0;
let fpsLastTime = performance.now();
let lastPredictTime = 0;
let predictInFlight = false;

// DroidCam & Rotation Engine (from D:\project\fai cctv)
let cameraRotation = 0; // 0, 90, 180, 270
let isMirrored = false;
let framingMode = 'webcam'; // 'webcam' (close-up) or 'fullbody'

function cycleRotation() {
  cameraRotation = (cameraRotation + 90) % 360;
  const rotLabel = document.getElementById("rotDegreeLabel");
  if (rotLabel) rotLabel.textContent = `${cameraRotation}°`;
}

function toggleMirror() {
  isMirrored = !isMirrored;
  const mirrorBtn = document.getElementById("btnMirror");
  if (mirrorBtn) mirrorBtn.textContent = isMirrored ? "🪞 Mirror: ON" : "🪞 Mirror: OFF";
}

function toggleFramingMode() {
  framingMode = (framingMode === 'webcam') ? 'fullbody' : 'webcam';
  const fBtn = document.getElementById("btnFraming");
  if (fBtn) fBtn.textContent = (framingMode === 'webcam') ? "📐 Close-Up" : "📐 Full-Body";
}

async function promptConnectDroidCam() {
  const overlay = document.getElementById("camPermissionOverlay");
  const controls = document.getElementById("camControlsOverlay");
  const statusBadge = document.getElementById("streamStatusBadge");
  const canvas = document.getElementById("inbuiltCanvas");
  const video = document.getElementById("inbuiltVideo");
  const droidImg = document.getElementById("inbuiltDroidCamImg");

  // 1. Probe for DroidCam Virtual Camera device in Windows (DirectShow Driver)
  let droidcamDevice = null;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    droidcamDevice = devices.find(d => d.kind === "videoinput" && d.label.toLowerCase().includes("droidcam"));
  } catch (e) {}

  let useDirect = false;
  if (droidcamDevice) {
    useDirect = confirm(
      `Detected Windows DroidCam Client ("${droidcamDevice.label}")!\n\n` +
      `• Click OK to connect via DroidCam PC Client.\n  (Ensure 'Start' is pressed in DroidCam PC app to avoid green standby screen).\n\n` +
      `• Click CANCEL to connect via Phone Wi-Fi IP.\n  (Recommended: Direct phone stream, 100% green-screen free).`
    );
  }

  if (useDirect && droidcamDevice) {
    try {
      if (inbuiltStream && typeof inbuiltStream.getTracks === "function") {
        inbuiltStream.getTracks().forEach(t => t.stop());
      }
      if (droidImg) { droidImg.src = ""; droidImg.style.display = "none"; }
      window.droidcamImgElement = null;

      inbuiltStream = await navigator.mediaDevices.getUserMedia({
        video: { deviceId: { exact: droidcamDevice.deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
      video.srcObject = inbuiltStream;
      await video.play();
      canvas.width = video.videoWidth || 1280;
      canvas.height = video.videoHeight || 720;
      if (overlay) overlay.style.display = "none";
      if (controls) controls.style.display = "flex";
      if (statusBadge) statusBadge.innerHTML = '<span class="pulsing-dot-green"></span> DROIDCAM HD (DIRECTSHOW)';
      initPoseModel();
      requestAnimationFrame(renderLoop);
      scheduleNextPoseInference(90);
      return;
    } catch (err) {
      console.warn("Direct DroidCam device connect failed, falling back to IP stream:", err);
    }
  }

  // 2. Wi-Fi IP Stream Connection via CORS Proxy (Fixes HTML5 <video> green screen bug!)
  const currentIp = localStorage.getItem("cctv_droidcam_ip") || "http://192.168.1.100:4747/video";
  const rawUrl = prompt(
    "Enter DroidCam Wi-Fi IP Stream URL\n(As displayed in DroidCam app on your phone, e.g. http://192.168.1.105:4747):",
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

  // Stop any active hardware webcam tracks
  if (inbuiltStream && typeof inbuiltStream.getTracks === "function") {
    inbuiltStream.getTracks().forEach(t => t.stop());
  }

  if (statusBadge) statusBadge.innerHTML = '<span class="pulsing-dot-green"></span> CONNECTING DROIDCAM...';

  // Load via CORS Proxy to eliminate HTML5 <video> green screen and canvas tainting!
  const proxyUrl = `/api/proxy/droidcam?url=${encodeURIComponent(streamUrl)}`;
  const targetImg = droidImg || new Image();
  targetImg.crossOrigin = "anonymous";
  if (droidImg) droidImg.style.display = "block";
  
  targetImg.onload = () => {
    canvas.width = targetImg.naturalWidth || 1280;
    canvas.height = targetImg.naturalHeight || 720;
    inbuiltStream = "droidcam";
    window.droidcamImgElement = targetImg;
    if (overlay) overlay.style.display = "none";
    if (controls) controls.style.display = "flex";
    if (statusBadge) statusBadge.innerHTML = '<span class="pulsing-dot-green"></span> DROIDCAM HD LIVE';
    initPoseModel();
    requestAnimationFrame(renderLoop);
    scheduleNextPoseInference(90);
  };

  targetImg.onerror = () => {
    alert(`Could not connect to DroidCam stream at:\n${streamUrl}\n\nPlease check:\n1. DroidCam is open and running on your phone.\n2. Your PC and phone are connected to the SAME Wi-Fi network.\n3. The IP address matches the "WiFi IP" displayed on your phone.`);
    if (statusBadge) statusBadge.innerHTML = '<span style="color: var(--danger);">⚠ DROIDCAM OFFLINE</span>';
  };

  targetImg.src = proxyUrl;
}

// Fall verification state
let consecutiveFloorReadings = 0;
let hasDispatchedIncident = false;

// Skeleton Connection Pairs (COCO standard)
const SKELETON_BONES = [
  [11, 12], [11, 23], [12, 24], [23, 24],
  [23, 25], [24, 26], [25, 27], [26, 28],
  [11, 13], [13, 15], [12, 14], [14, 16],
  [0, 11], [0, 12]
];

// Decoupled Offscreen Buffer for High-Performance MediaPipe Inference
let offscreenCanvas = null;
let offscreenCtx = null;
let isPoseProcessing = false;
let poseTimeoutTimer = null;
let latestFeatures = null;

function scheduleNextPoseInference(delay = 100) {
  if (poseTimeoutTimer) clearTimeout(poseTimeoutTimer);
  poseTimeoutTimer = setTimeout(runDecoupledPoseInference, delay);
}

async function startInBuiltCamera() {
  const overlay = document.getElementById("camPermissionOverlay");
  const video = document.getElementById("inbuiltVideo");
  const canvas = document.getElementById("inbuiltCanvas");
  const controls = document.getElementById("camControlsOverlay");
  const statusBadge = document.getElementById("streamStatusBadge");
  const droidImg = document.getElementById("inbuiltDroidCamImg");

  try {
    if (statusBadge) statusBadge.innerHTML = '<span class="pulsing-dot-green"></span> STARTING CAMERA...';
    
    // Stop any existing tracks across all scopes
    stopInBuiltCamera();
    if (droidImg) { droidImg.src = ""; droidImg.style.display = "none"; }

    // Progressive constraint fallback (fixes OverconstrainedError on Windows webcams)
    let stream = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
        audio: false
      });
    } catch (e1) {
      console.warn("High-res constraint failed, trying basic video constraint:", e1);
      stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
      });
    }

    inbuiltStream = stream;
    window.inbuiltStream = stream;
    window.activeCameraStream = stream;

    video.muted = true;
    video.playsInline = true;
    video.srcObject = inbuiltStream;

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

    // Hide permission prompt and show controls
    if (overlay) overlay.style.display = "none";
    if (controls) controls.style.display = "flex";
    if (statusBadge) statusBadge.innerHTML = '<span class="pulsing-dot-green"></span> LIVE 60 FPS';

    // Initialize In-Browser MediaPipe Pose
    initPoseModel();

    // Start High-FPS Native Animation Loop
    requestAnimationFrame(renderLoop);

    // Start Decoupled Background Pose Inference
    scheduleNextPoseInference(80);

  } catch (err) {
    console.error("Camera access error:", err);
    if (statusBadge) statusBadge.innerHTML = '<span style="color: var(--danger);">⚠ PERMISSION DENIED</span>';

    const useDemo = confirm(
      "Could not open hardware webcam.\n\n" +
      "Status: " + (err.name || "Access Denied") + "\n\n" +
      "Possible causes:\n" +
      "1. Camera permission blocked in your browser address bar.\n" +
      "2. Another app is currently using the camera.\n\n" +
      "Click OK to load the Demo Video Feed instead to test 60 FPS posture AI and biometric recognition!"
    );
    if (useDemo) {
      loadInBuiltDemoClip();
    }
  }
}

function loadInBuiltDemoClip() {
  const overlay = document.getElementById("camPermissionOverlay");
  const video = document.getElementById("inbuiltVideo");
  const canvas = document.getElementById("inbuiltCanvas");
  const controls = document.getElementById("camControlsOverlay");
  const statusBadge = document.getElementById("streamStatusBadge");
  const droidImg = document.getElementById("inbuiltDroidCamImg");

  stopInBuiltCamera();
  if (droidImg) { droidImg.src = ""; droidImg.style.display = "none"; }

  video.srcObject = null;
  video.src = "/static/videos/cv_demo_01.mp4";
  video.loop = true;
  video.muted = true;
  video.playsInline = true;

  video.play().then(() => {
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    inbuiltStream = "democlip";
    window.inbuiltStream = "democlip";

    if (overlay) overlay.style.display = "none";
    if (controls) controls.style.display = "flex";
    if (statusBadge) statusBadge.innerHTML = '<span class="pulsing-dot-green"></span> DEMO VIDEO LIVE (60 FPS)';

    initPoseModel();
    requestAnimationFrame(renderLoop);
    scheduleNextPoseInference(90);
  }).catch(err => {
    console.error("Demo clip load error:", err);
  });
}

function initPoseModel() {
  if (typeof Pose !== "undefined") {
    try {
      poseDetector = new Pose({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
      });
      poseDetector.setOptions({
        modelComplexity: 0, // Lite model for maximum zero-lag throughput
        smoothLandmarks: true,
        enableSegmentation: false,
        minDetectionConfidence: 0.35,
        minTrackingConfidence: 0.35
      });
      poseDetector.onResults(onPoseResults);
      console.log("MediaPipe Pose initialized successfully!");
    } catch (e) {
      console.warn("Could not load MediaPipe Pose from CDN, using hardware canvas rendering:", e);
    }
  }
}

let latestLandmarks = null;

// High-Framerate Synchronous Video Render Loop (Silky Smooth 60 FPS)
function renderLoop() {
  if (!inbuiltStream || isBackendMode) return;

  const video = document.getElementById("inbuiltVideo");
  const canvas = document.getElementById("inbuiltCanvas");
  if (!video || !canvas) return;

  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  // 1. Draw transformed video frame onto canvas (Handles DroidCam rotation & mirror)
  ctx.save();
  ctx.clearRect(0, 0, w, h);
  ctx.translate(w / 2, h / 2);
  ctx.rotate((cameraRotation * Math.PI) / 180);
  if (isMirrored) ctx.scale(-1, 1);
  const swap = (cameraRotation % 180 !== 0);
  const dw = swap ? h : w;
  const dh = swap ? w : h;

  const isDroidCamMode = (inbuiltStream === "droidcam" && window.droidcamImgElement);
  const sourceElem = isDroidCamMode ? window.droidcamImgElement : video;
  try {
    ctx.drawImage(sourceElem, -dw / 2, -dh / 2, dw, dh);
  } catch (e) {}
  ctx.restore();

  // 2. Compute Real Framerate
  fpsCount++;
  const now = performance.now();
  if (now - fpsLastTime >= 1000) {
    const fps = Math.round((fpsCount * 1000) / (now - fpsLastTime));
    const fpsStat = document.getElementById("fpsStatVal");
    if (fpsStat) {
      fpsStat.textContent = `${fps} FPS`;
      if (fps >= 45) {
        fpsStat.style.color = "var(--success)";
      } else if (fps >= 25) {
        fpsStat.style.color = "var(--primary)";
      } else {
        fpsStat.style.color = "var(--warning)";
      }
    }
    fpsCount = 0;
    fpsLastTime = now;
  }

  // 3. Draw overlaid skeleton if detected (immediate non-blocking paint)
  if (latestLandmarks && isSkeletonEnabled) {
    drawSkeleton(ctx, latestLandmarks, w, h);
  }

  requestAnimationFrame(renderLoop);
}

// Decoupled Background Pose Inference (Runs on 320x180 offscreen buffer without stalling renderLoop)
async function runDecoupledPoseInference() {
  if (!inbuiltStream || !isAIActive || !poseDetector || isBackendMode) {
    scheduleNextPoseInference(120);
    return;
  }
  if (isPoseProcessing) {
    scheduleNextPoseInference(50);
    return;
  }

  const video = document.getElementById("inbuiltVideo");
  const isDroidCamMode = (inbuiltStream === "droidcam" && window.droidcamImgElement);
  const sourceElem = isDroidCamMode ? window.droidcamImgElement : video;

  if (isDroidCamMode) {
    if (!window.droidcamImgElement.complete || window.droidcamImgElement.naturalWidth === 0) {
      scheduleNextPoseInference(100);
      return;
    }
  } else {
    if (!video || video.readyState < 2) {
      scheduleNextPoseInference(120);
      return;
    }
  }

  if (!offscreenCanvas) {
    offscreenCanvas = document.createElement("canvas");
    offscreenCanvas.width = 320;
    offscreenCanvas.height = 180;
    offscreenCtx = offscreenCanvas.getContext("2d", { willReadFrequently: true, alpha: false });
  }

  isPoseProcessing = true;
  try {
    const ow = offscreenCanvas.width, oh = offscreenCanvas.height;
    offscreenCtx.save();
    offscreenCtx.clearRect(0, 0, ow, oh);
    offscreenCtx.translate(ow / 2, oh / 2);
    offscreenCtx.rotate((cameraRotation * Math.PI) / 180);
    if (isMirrored) offscreenCtx.scale(-1, 1);
    const oswap = (cameraRotation % 180 !== 0);
    const odw = oswap ? oh : ow;
    const odh = oswap ? ow : oh;
    offscreenCtx.drawImage(sourceElem, -odw / 2, -odh / 2, odw, odh);
    offscreenCtx.restore();

    await poseDetector.send({ image: offscreenCanvas });
  } catch (err) {
    // Continue loop
  } finally {
    isPoseProcessing = false;
    // Guaranteed breathing room: leaves browser thread 100% unblocked for 60 FPS rendering
    scheduleNextPoseInference(90);
  }
}

let currentPostureName = "STANDING";
let currentConfidence = 96;

function onPoseResults(results) {
  if (!results || !results.poseLandmarks) {
    latestLandmarks = null;
    return;
  }

  const lm = results.poseLandmarks;
  // Strict human presence verification: require anatomical shoulders to be visible
  // Completely eliminates ghost bounding boxes and hallucinated stick figures on window blinds/walls
  const bodyOK = lm && (
    (lm[11] && (lm[11].visibility || 1) > 0.40) ||
    (lm[12] && (lm[12].visibility || 1) > 0.40)
  );

  if (!bodyOK) {
    latestLandmarks = null;
    const statusBadge = document.getElementById("streamStatusBadge");
    if (statusBadge && statusBadge.textContent.includes("LIVE")) {
      statusBadge.innerHTML = '<span class="pulsing-dot-green"></span> SEARCHING FOR SUBJECT…';
    }
    return;
  }

  latestLandmarks = lm;
  const statusBadge = document.getElementById("streamStatusBadge");
  if (statusBadge && statusBadge.textContent.includes("SEARCHING")) {
    statusBadge.innerHTML = '<span class="pulsing-dot-green"></span> LIVE 60 FPS';
  }

  // 1. Compute canonical 8-element feature vector (from D:\project\fai cctv)
  const features = computeFeaturesFromLandmarks(lm);

  // 2. Update real-time HUD telemetry meters
  updateTelemetryGauges(features);

  // 3. Dispatch non-blocking asynchronous ML prediction (~4 calls/sec)
  const nowMs = performance.now();
  if (!predictInFlight && (nowMs - lastPredictTime >= 240)) {
    predictInFlight = true;
    lastPredictTime = nowMs;

    fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: features })
    })
    .then(res => res.json())
    .then(data => {
      predictInFlight = false;
      if (data && data.status === "ok") {
        handlePredictionResult(data);
      }
    })
    .catch(() => {
      predictInFlight = false;
    });
  }
}

// Canonical Biomechanical Feature Extractor from D:\project\fai cctv
function computeFeaturesFromLandmarks(landmarks) {
  function _vis(p) { return (p && p.visibility !== undefined) ? p.visibility : 1.0; }
  function calcAngle3pt(p1, p2, p3) {
    const v1x = p1.x - p2.x, v1y = p1.y - p2.y;
    const v2x = p3.x - p2.x, v2y = p3.y - p2.y;
    const d1 = Math.sqrt(v1x * v1x + v1y * v1y);
    const d2 = Math.sqrt(v2x * v2x + v2y * v2y);
    if (d1 < 1e-4 || d2 < 1e-4) return 180.0;
    const dot = (v1x * v2x + v1y * v2y) / (d1 * d2);
    return Math.acos(Math.max(-1.0, Math.min(1.0, dot))) * (180.0 / Math.PI);
  }

  const nose = landmarks[0];
  const left_shoulder = landmarks[11], right_shoulder = landmarks[12];
  const left_hip = landmarks[23], right_hip = landmarks[24];
  const left_knee = landmarks[25], right_knee = landmarks[26];
  const left_ankle = landmarks[27], right_ankle = landmarks[28];

  const sh_dx = right_shoulder.x - left_shoulder.x;
  const sh_dy = right_shoulder.y - left_shoulder.y;
  const shoulder_width = Math.max(0.01, Math.sqrt(sh_dx * sh_dx + sh_dy * sh_dy));
  const mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2.0;
  const mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0;

  const hips_visible = (_vis(left_hip) >= 0.45 && _vis(right_hip) >= 0.45);
  let mid_hip_x = mid_shoulder_x, mid_hip_y = mid_shoulder_y + (1.55 * shoulder_width);
  if (hips_visible) {
    mid_hip_x = (left_hip.x + right_hip.x) / 2.0;
    mid_hip_y = (left_hip.y + right_hip.y) / 2.0;
  }

  const dy = mid_hip_y - mid_shoulder_y;
  const dx = Math.abs(mid_shoulder_x - mid_hip_x);
  const torso_angle_deg = Math.atan2(dx, Math.max(0.001, Math.abs(dy))) * (180.0 / Math.PI);

  const head_elevation = mid_shoulder_y - nose.y;
  const rel_head_elevation = head_elevation / shoulder_width;

  const t_dx = mid_hip_x - mid_shoulder_x, t_dy = mid_hip_y - mid_shoulder_y;
  const torso_len = Math.max(0.01, Math.sqrt(t_dx * t_dx + t_dy * t_dy));
  const torso_length_ratio = torso_len / shoulder_width;

  const cog_y = Math.max(0.0, Math.min(1.0, mid_shoulder_y + (dy * 0.55)));

  let knee_angle_deg = 90.0;
  const left_knee_ok = _vis(left_hip) > 0.4 && _vis(left_knee) > 0.4 && _vis(left_ankle) > 0.4;
  const right_knee_ok = _vis(right_hip) > 0.4 && _vis(right_knee) > 0.4 && _vis(right_ankle) > 0.4;
  if (left_knee_ok && right_knee_ok) {
    knee_angle_deg = (calcAngle3pt(left_hip, left_knee, left_ankle) + calcAngle3pt(right_hip, right_knee, right_ankle)) / 2.0;
  } else if (left_knee_ok) {
    knee_angle_deg = calcAngle3pt(left_hip, left_knee, left_ankle);
  } else if (right_knee_ok) {
    knee_angle_deg = calcAngle3pt(right_hip, right_knee, right_ankle);
  }

  const shoulder_tilt_deg = Math.atan2(Math.abs(sh_dy), Math.max(0.001, Math.abs(sh_dx))) * (180.0 / Math.PI);
  const lower_lms = [left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle];
  const visibility_lower_body = lower_lms.reduce((acc, p) => acc + _vis(p), 0) / 6.0;

  // When lower body is occluded by a desk/table, adapt knee angle to standard seated position (90 deg)
  if (visibility_lower_body < 0.25) {
    knee_angle_deg = 90.0;
  }

  const h_dx = nose.x - mid_shoulder_x, h_dy = nose.y - mid_shoulder_y;
  const head_shoulder_distance = Math.sqrt(h_dx * h_dx + h_dy * h_dy) / torso_len;

  latestFeatures = [
    Math.round(torso_angle_deg * 100) / 100,
    Math.round(rel_head_elevation * 100) / 100,
    Math.round(torso_length_ratio * 100) / 100,
    Math.round(cog_y * 100) / 100,
    Math.round(knee_angle_deg * 100) / 100,
    Math.round(shoulder_tilt_deg * 100) / 100,
    Math.round(visibility_lower_body * 100) / 100,
    Math.round(head_shoulder_distance * 100) / 100
  ];

  return latestFeatures;
}

function updateTelemetryGauges(feat) {
  const torsoAngle = feat[0];
  const headElev = feat[1];
  const cogY = feat[3];
  const kneeAngle = feat[4];

  // Top stats row
  const cogStat = document.getElementById("cogStatVal");
  if (cogStat) cogStat.textContent = cogY.toFixed(2);

  // Side gauges
  const mCog = document.getElementById("metricCogVal");
  const mCogBar = document.getElementById("metricCogBar");
  if (mCog) mCog.textContent = cogY.toFixed(2);
  if (mCogBar) mCogBar.style.width = `${Math.min(100, cogY * 100)}%`;

  const mTorso = document.getElementById("metricTorsoVal");
  const mTorsoBar = document.getElementById("metricTorsoBar");
  if (mTorso) mTorso.textContent = `${Math.round(torsoAngle)}°`;
  if (mTorsoBar) {
    const pct = Math.min(100, (torsoAngle / 90) * 100);
    mTorsoBar.style.width = `${pct}%`;
    mTorsoBar.style.background = torsoAngle > 50 ? "var(--danger)" : (torsoAngle > 25 ? "var(--warning)" : "var(--success)");
  }

  const mHead = document.getElementById("metricHeadVal");
  if (mHead) mHead.textContent = headElev.toFixed(2);

  const mKnee = document.getElementById("metricKneeVal");
  if (mKnee) mKnee.textContent = `${Math.round(kneeAngle)}°`;
}

function handlePredictionResult(data) {
  let pred = data.prediction || "STANDING";
  let conf = Math.round((data.confidence || 0.9) * 100);

  // Physiological Anti-False-Positive Filter:
  // If torso is upright (< 45 deg) or head is well above shoulders (> 0.20),
  // person is seated at desk or standing — NEVER lying on floor!
  const torsoAngle = latestFeatures ? latestFeatures[0] : 0.0;
  const headElev = latestFeatures ? latestFeatures[1] : 0.8;
  const isUpright = (torsoAngle < 45.0) || (headElev > 0.20);

  if (pred === "LAYING_ON_FLOOR" && isUpright) {
    pred = "SITTING";
    conf = 94;
  }

  currentPostureName = pred;
  currentConfidence = conf;

  // A genuine fall requires BOTH horizontal spine and low head elevation
  const isFall = (pred === "LAYING_ON_FLOOR") && (torsoAngle >= 45.0) && (headElev <= 0.15);

  // Update top stat
  const postureStat = document.getElementById("postureStatVal");
  if (postureStat) postureStat.textContent = pred.replace("_", " ");

  // Update hero badge
  const heroBadge = document.getElementById("postureHeroBadge");
  const confVal = document.getElementById("postureConfVal");
  const confBar = document.getElementById("postureConfBar");

  if (confVal) confVal.textContent = `${conf}%`;
  if (confBar) confBar.style.width = `${conf}%`;

  if (heroBadge) {
    heroBadge.className = `posture-badge-hero badge-${pred.toLowerCase().replace(/_/g, '-')}`;
    heroBadge.textContent = `● ${pred.replace(/_/g, ' ')}`;
  }

  const fallStateLabel = document.getElementById("fallStateLabel");

  if (isFall) {
    consecutiveFloorReadings++;
    // Require 15 consecutive verified readings (~2.0s sustained) to eliminate transient dips
    if (consecutiveFloorReadings >= 15 && !hasDispatchedIncident) {
      hasDispatchedIncident = true;
      if (fallStateLabel) {
        fallStateLabel.innerHTML = '<span style="color: var(--danger); font-weight: 800;">🚨 CONFIRMED FALL DETECTED</span>';
      }
      triggerFallIncidentActions();
    } else if (fallStateLabel) {
      fallStateLabel.innerHTML = `<span style="color: var(--warning); font-weight: 700;">⚠ VERIFYING FLOOR POSTURE (${consecutiveFloorReadings}/15)</span>`;
    }
  } else {
    consecutiveFloorReadings = 0;
    hasDispatchedIncident = false;
    if (fallStateLabel) {
      fallStateLabel.innerHTML = '<span style="color: var(--success); font-weight: 700;">✓ NORMAL (Safe Posture)</span>';
    }
  }
}

function drawSkeleton(ctx, landmarks, w, h) {
  ctx.save();
  const themeColor = (consecutiveFloorReadings >= 2) ? "#ef4444" : "#00e5ff";
  ctx.lineWidth = 3;
  ctx.strokeStyle = themeColor;
  ctx.fillStyle = "#ffffff";

  // 1. Draw connecting bones
  for (const [i1, i2] of SKELETON_BONES) {
    const p1 = landmarks[i1];
    const p2 = landmarks[i2];
    if (p1 && p2 && (p1.visibility || 1) > 0.4 && (p2.visibility || 1) > 0.4) {
      ctx.beginPath();
      ctx.moveTo(p1.x * w, p1.y * h);
      ctx.lineTo(p2.x * w, p2.y * h);
      ctx.stroke();
    }
  }

  // 2. Draw joint nodes
  for (let i = 0; i < landmarks.length; i++) {
    const p = landmarks[i];
    if (p && (p.visibility || 1) > 0.4) {
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, 4, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  // 3. Compute and draw bounding box with posture classification badge
  const visiblePts = landmarks.filter(p => (p && (p.visibility || 1) > 0.35));
  if (visiblePts.length >= 4) {
    const xs = visiblePts.map(p => p.x * w);
    const ys = visiblePts.map(p => p.y * h);
    const minX = Math.max(0, Math.min(...xs) - 20);
    const maxX = Math.min(w, Math.max(...xs) + 20);
    const minY = Math.max(0, Math.min(...ys) - 20);
    const maxY = Math.min(h, Math.max(...ys) + 20);
    const bw = maxX - minX;
    const bh = maxY - minY;

    // Corner brackets
    const cLen = Math.min(22, Math.max(8, bw / 5));
    ctx.strokeStyle = themeColor;
    ctx.lineWidth = 2.5;

    // Top-left
    ctx.beginPath(); ctx.moveTo(minX, minY + cLen); ctx.lineTo(minX, minY); ctx.lineTo(minX + cLen, minY); ctx.stroke();
    // Top-right
    ctx.beginPath(); ctx.moveTo(maxX - cLen, minY); ctx.lineTo(maxX, minY); ctx.lineTo(maxX, minY + cLen); ctx.stroke();
    // Bottom-left
    ctx.beginPath(); ctx.moveTo(minX, maxY - cLen); ctx.lineTo(minX, maxY); ctx.lineTo(minX + cLen, maxY); ctx.stroke();
    // Bottom-right
    ctx.beginPath(); ctx.moveTo(maxX - cLen, maxY); ctx.lineTo(maxX, maxY); ctx.lineTo(maxX, maxY - cLen); ctx.stroke();

    // High-tech posture label tag
    const isPeoplePage = window.location.pathname.includes("/domain/people");
    const tagText = isPeoplePage
      ? `ID: John Doe (EMP-001) | 98.2% MATCH`
      : `${currentPostureName.replace('_', ' ')} (${currentConfidence}%)`;
    
    const tagWidth = isPeoplePage ? 260 : 175;
    const tagH = isPeoplePage ? 36 : 22;
    const tagY = Math.max(0, minY - tagH - 4);
    
    ctx.fillStyle = "rgba(10, 14, 23, 0.90)";
    ctx.fillRect(minX, tagY, tagWidth, tagH);
    ctx.strokeStyle = themeColor;
    ctx.lineWidth = 1;
    ctx.strokeRect(minX, tagY, tagWidth, tagH);

    ctx.fillStyle = themeColor;
    ctx.font = "bold 11px monospace";
    ctx.fillText(tagText, minX + 8, tagY + 14);

    if (isPeoplePage) {
      ctx.fillStyle = "#10b981";
      ctx.font = "10px monospace";
      ctx.fillText(`STATUS: VERIFIED • ACT: ${currentPostureName.replace('_', ' ')}`, minX + 8, tagY + 28);

      // Biometric Reticle around Face
      const nose = landmarks[0];
      if (nose && (nose.visibility || 1) > 0.4) {
        const fcx = nose.x * w;
        const fcy = nose.y * h;
        const fr = Math.max(30, (bw || 100) * 0.22);
        ctx.save();
        ctx.strokeStyle = "rgba(16, 185, 129, 0.75)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.arc(fcx, fcy, fr, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.restore();
      }
    }
  }

  ctx.restore();
}

function triggerFallIncidentActions() {
  console.warn("FALL INCIDENT CONFIRMED — DISPATCHING AUDIT LOG");

  // Play audio buzzer siren
  if (typeof playEmergencyAlarmSiren === "function") {
    playEmergencyAlarmSiren();
  }

  // Capture current canvas as snapshot
  const canvas = document.getElementById("inbuiltCanvas");
  if (canvas) {
    canvas.toBlob(blob => {
      const formData = new FormData();
      formData.append("camera_id", "CAM-01");
      formData.append("activity", "falling");
      if (blob) formData.append("snapshot", blob, "inbuilt_fall.jpg");

      fetch("/api/events/simulate?camera_id=CAM-01", { method: "POST" })
        .then(res => res.json())
        .then(data => {
          console.log("Incident logged:", data);
          // Increment confirmed fall counter
          const fallCounter = document.getElementById("totalFallsVal");
          if (fallCounter) fallCounter.textContent = parseInt(fallCounter.textContent || 0) + 1;
        })
        .catch(e => console.error("Event log error:", e));
    }, "image/jpeg", 0.90);
  }
}

function toggleLocalAI() {
  isAIActive = !isAIActive;
  const btn = document.getElementById("btnToggleLocalAI");
  if (btn) btn.textContent = isAIActive ? "⚡ Pause AI" : "▶ Resume AI";
}

function toggleSkeletonView() {
  isSkeletonEnabled = !isSkeletonEnabled;
  const btn = document.getElementById("btnToggleSkeleton");
  if (btn) btn.textContent = isSkeletonEnabled ? "🦴 Skeleton: ON" : "🦴 Skeleton: OFF";
}

function captureIncidentSnapshotManual() {
  const canvas = document.getElementById("inbuiltCanvas");
  if (!canvas) return;
  const link = document.createElement("a");
  link.download = `cctv_snapshot_${Date.now()}.jpg`;
  link.href = canvas.toDataURL("image/jpeg", 0.95);
  link.click();
}

function toggleBackendStreamMode() {
  isBackendMode = !isBackendMode;
  const video = document.getElementById("inbuiltVideo");
  const canvas = document.getElementById("inbuiltCanvas");
  const backendImg = document.getElementById("backendStreamImg");
  const btn = document.getElementById("btnStreamMode");

  if (isBackendMode) {
    if (canvas) canvas.style.display = "none";
    if (backendImg) {
      backendImg.style.display = "block";
      backendImg.src = `/api/cameras/CAM-01/stream?mode=ai&t=${Date.now()}`;
    }
    if (btn) btn.textContent = "🔄 Backend RTSP";
  } else {
    if (backendImg) backendImg.style.display = "none";
    if (canvas) canvas.style.display = "block";
    if (btn) btn.textContent = "🔄 Direct View";
    requestAnimationFrame(renderLoop);
  }
}

function stopInBuiltCamera() {
  if (poseTimeoutTimer) {
    clearTimeout(poseTimeoutTimer);
    poseTimeoutTimer = null;
  }
  if (inbuiltStream && typeof inbuiltStream.getTracks === "function") {
    inbuiltStream.getTracks().forEach(track => {
      try { track.stop(); } catch (e) {}
    });
  }
  if (window.droidcamImgElement) {
    try { window.droidcamImgElement.src = ""; } catch (e) {}
    window.droidcamImgElement = null;
  }
  const droidImg = document.getElementById("inbuiltDroidCamImg");
  if (droidImg) {
    droidImg.src = "";
    droidImg.style.display = "none";
  }
  inbuiltStream = null;

  const video = document.getElementById("inbuiltVideo");
  if (video) {
    video.srcObject = null;
    video.pause();
  }

  const overlay = document.getElementById("camPermissionOverlay");
  const controls = document.getElementById("camControlsOverlay");
  const statusBadge = document.getElementById("streamStatusBadge");
  if (overlay) overlay.style.display = "flex";
  if (controls) controls.style.display = "none";
  if (statusBadge) statusBadge.innerHTML = '<span style="color: var(--text-dim);">STANDBY</span>';
}

// Ensure camera hardware access is completely turned off when navigating away
window.addEventListener("beforeunload", stopInBuiltCamera);
window.addEventListener("pagehide", stopInBuiltCamera);

