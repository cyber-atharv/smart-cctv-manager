// Real-time CCTV Safety Dashboard Client
let socket = null;
let currentAlertEvent = null;
const streamModes = {}; // cameraId -> "ai" | "raw"

function initAlertWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/alerts`;
  
  socket = new WebSocket(wsUrl);

  socket.onopen = () => {
    console.log("[WebSocket] Connected to CCTV Safety Alert Dispatcher.");
  };

  socket.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (payload.type === "SAFETY_ALERT") {
        onSafetyAlertReceived(payload.data);
      }
    } catch (e) {
      console.error("[WebSocket] Message parsing error:", e);
    }
  };

  socket.onclose = () => {
    console.warn("[WebSocket] Disconnected. Reconnecting in 3s...");
    setTimeout(initAlertWebSocket, 3000);
  };
}

function onSafetyAlertReceived(eventData) {
  console.warn("[SAFETY INCIDENT ALERT]", eventData);
  currentAlertEvent = eventData;

  // Populate alert modal fields
  const elEventId = document.getElementById("modalEventId");
  if (elEventId) elEventId.innerText = eventData.id || "EVT-ALERT";
  
  const elCam = document.getElementById("modalCamera");
  if (elCam) elCam.innerText = eventData.camera_name || eventData.camera_id;
  
  const elTrack = document.getElementById("modalTrack");
  if (elTrack) elTrack.innerText = `Track #${eventData.track_id || 'N/A'}`;
  
  const empBadge = document.getElementById("modalEmployee");
  if (empBadge) {
    if (eventData.identity_status === "recognized" && eventData.employee_code) {
      empBadge.innerHTML = `<span class="badge badge-success">${eventData.employee_code} (${eventData.employee_name})</span>`;
    } else {
      empBadge.innerHTML = `<span class="badge badge-unknown">Unknown Subject</span>`;
    }
  }

  const elAct = document.getElementById("modalActivity");
  if (elAct) {
    elAct.innerText = `${(eventData.activity || 'FALL').toUpperCase()} (${Math.round((eventData.activity_confidence || 0.95) * 100)}%)`;
  }
  
  const elTs = document.getElementById("modalTimestamp");
  if (elTs) elTs.innerText = eventData.timestamp || new Date().toLocaleTimeString();
  
  const snapshotImg = document.getElementById("modalSnapshot");
  if (snapshotImg) {
    if (eventData.snapshot_path) {
      snapshotImg.src = eventData.snapshot_path;
      snapshotImg.style.display = "block";
    } else {
      snapshotImg.style.display = "none";
    }
  }

  // Open modal
  const modal = document.getElementById("safetyAlertModal");
  if (modal) {
    modal.classList.add("active");
  }

  // Prepend to table
  prependEventToTable(eventData);
}

function closeAlertModal() {
  const modal = document.getElementById("safetyAlertModal");
  if (modal) modal.classList.remove("active");
  currentAlertEvent = null;
}

async function reviewCurrentEvent(decision) {
  if (!currentAlertEvent || !currentAlertEvent.id) {
    closeAlertModal();
    return;
  }

  try {
    const res = await fetch(`/api/events/${currentAlertEvent.id}/${decision}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" }
    });
    const data = await res.json();
    if (data.success) {
      alert(`Event ${currentAlertEvent.id} marked as ${decision.toUpperCase()}`);
      closeAlertModal();
      window.location.reload();
    }
  } catch (err) {
    console.error("Error submitting review:", err);
    alert("Failed to update event review.");
  }
}

async function triggerSimulatedFall() {
  try {
    const res = await fetch('/api/events/simulate', {
      method: "POST",
      headers: { "Content-Type": "application/json" }
    });
    const data = await res.json();
    if (data.success) {
      console.log("Simulated fall event dispatched:", data.event);
    }
  } catch (e) {
    console.error("Simulation error:", e);
  }
}

function prependEventToTable(evt) {
  const tableBody = document.getElementById("recentEventsTableBody");
  if (!tableBody) return;

  const row = document.createElement("tr");
  const isRec = evt.identity_status === "recognized";
  const empText = isRec ? `${evt.employee_code} (${evt.employee_name})` : 'Unknown';
  const badgeClass = isRec ? 'badge-success' : 'badge-unknown';

  row.innerHTML = `
    <td><strong>${evt.id}</strong></td>
    <td>${evt.camera_name || evt.camera_id}</td>
    <td><span class="badge ${badgeClass}">${empText}</span></td>
    <td><span class="badge badge-danger">${evt.activity.toUpperCase()} (${Math.round((evt.activity_confidence || 0.9) * 100)}%)</span></td>
    <td>${evt.timestamp || new Date().toLocaleTimeString()}</td>
    <td><span class="badge badge-warning">${evt.status}</span></td>
    <td>
      <a href="/events" class="btn btn-outline" style="padding: 4px 8px; font-size: 11px;">Review</a>
    </td>
  `;
  tableBody.insertBefore(row, tableBody.firstChild);
}

// ==================== REAL CAMERA CONTROLS ====================

async function toggleCameraAI(cameraId) {
  try {
    const res = await fetch(`/api/cameras/${cameraId}/toggle-ai`, { method: "POST" });
    const data = await res.json();
    if (data.success) {
      const isAi = data.ai_enabled;
      // Update tags
      const aiTag = document.getElementById(`ai-tag-${cameraId}`);
      if (aiTag) {
        aiTag.className = `badge ${isAi ? 'badge-ai-active' : 'badge-ai-stopped'}`;
        aiTag.innerText = isAi ? "● AI ACTIVE" : "○ AI STOPPED";
      }
      const fleetAiTag = document.getElementById(`fleet-ai-badge-${cameraId}`);
      if (fleetAiTag) {
        fleetAiTag.className = `badge ${isAi ? 'badge-ai-active' : 'badge-ai-stopped'}`;
        fleetAiTag.innerText = isAi ? "AI: ACTIVE" : "AI: STOPPED";
      }

      // Update button text
      const btnText = document.getElementById(`btn-text-ai-${cameraId}`);
      if (btnText) btnText.innerText = isAi ? "Pause AI" : "Start AI";
      const fleetBtnText = document.getElementById(`fleet-ai-btn-text-${cameraId}`);
      if (fleetBtnText) fleetBtnText.innerText = isAi ? "Pause AI" : "Start AI";
    }
  } catch (err) {
    console.error("Error toggling camera AI:", err);
  }
}

function toggleStreamMode(cameraId) {
  const currentMode = streamModes[cameraId] || "ai";
  const newMode = (currentMode === "ai") ? "raw" : "ai";
  streamModes[cameraId] = newMode;

  const streamImg = document.getElementById(`stream-${cameraId}`) || document.getElementById(`fleet-stream-${cameraId}`);
  if (streamImg) {
    streamImg.src = `/api/cameras/${cameraId}/stream?mode=${newMode}&t=${Date.now()}`;
  }

  const btnText = document.getElementById(`btn-text-mode-${cameraId}`);
  if (btnText) btnText.innerText = (newMode === "ai") ? "Raw View" : "AI View";
}

async function reconnectCamera(cameraId) {
  try {
    const res = await fetch(`/api/cameras/${cameraId}/reconnect`, { method: "POST" });
    const data = await res.json();
    if (data.success) {
      const streamImg = document.getElementById(`stream-${cameraId}`) || document.getElementById(`fleet-stream-${cameraId}`);
      if (streamImg) {
        streamImg.src = `/api/cameras/${cameraId}/stream?mode=${streamModes[cameraId] || 'ai'}&t=${Date.now()}`;
      }
      setTimeout(() => window.location.reload(), 1000);
    }
  } catch (err) {
    console.error("Error reconnecting camera:", err);
  }
}

async function deleteCamera(cameraId) {
  if (!confirm(`Are you sure you want to remove camera ${cameraId}?`)) return;
  try {
    const res = await fetch(`/api/cameras/${cameraId}`, { method: "DELETE" });
    const data = await res.json();
    if (data.success) {
      window.location.reload();
    }
  } catch (err) {
    console.error("Error deleting camera:", err);
    alert("Failed to delete camera.");
  }
}

// ==================== ADD CAMERA MODAL & VERIFICATION ====================

function openAddCameraModal() {
  const modal = document.getElementById("addCameraModal");
  if (modal) {
    modal.classList.add("active");
    selectSourceTab("webcam");
  }
}

function closeAddCameraModal() {
  const modal = document.getElementById("addCameraModal");
  if (modal) modal.classList.remove("active");
  const vBox = document.getElementById("verificationBox");
  if (vBox) vBox.style.display = "none";
}

function selectSourceTab(type) {
  document.getElementById("camSourceType").value = type;
  document.querySelectorAll(".source-tab").forEach(tab => {
    tab.classList.toggle("active", tab.getAttribute("data-type") === type);
  });

  const uriInput = document.getElementById("camUri");
  const uriLabel = document.getElementById("uriLabel");
  const uriHelper = document.getElementById("uriHelper");
  const scanBtn = document.getElementById("scanLocalBtn");
  const rtspBox = document.getElementById("rtspCredentialsBox");

  if (type === "webcam") {
    uriLabel.innerText = "Local Device Index";
    uriInput.placeholder = "0 (Integrated) or 1 (USB/DroidCam)";
    uriInput.value = "0";
    uriHelper.innerHTML = "Enter <code>0</code> for primary laptop webcam, <code>1</code> for external USB camera.";
    scanBtn.style.display = "block";
    rtspBox.style.display = "none";
  } else if (type === "mobile") {
    uriLabel.innerText = "Mobile Stream URL or Device Index";
    uriInput.placeholder = "http://192.168.1.100:4747/video (or 1 for USB)";
    uriInput.value = "http://192.168.1.100:4747/video";
    uriHelper.innerHTML = "For <strong>DroidCam</strong> WiFi: <code>http://&lt;phone-ip&gt;:4747/video</code>. For <strong>IP Webcam</strong>: <code>http://&lt;phone-ip&gt;:8080/video</code>.";
    scanBtn.style.display = "none";
    rtspBox.style.display = "none";
  } else if (type === "rtsp") {
    uriLabel.innerText = "RTSP IP Camera Stream URL";
    uriInput.placeholder = "rtsp://192.168.1.50:554/live/ch0";
    uriInput.value = "rtsp://192.168.1.50:554/live/ch0";
    uriHelper.innerHTML = "Standard RTSP endpoint: <code>rtsp://&lt;ip&gt;:554/&lt;path&gt;</code>.";
    scanBtn.style.display = "none";
    rtspBox.style.display = "grid";
  } else if (type === "file") {
    uriLabel.innerText = "Local Video File Path";
    uriInput.placeholder = "sample_cctv.mp4";
    uriInput.value = "sample_fall.mp4";
    uriHelper.innerHTML = "Relative path or absolute file path to MP4/AVI file.";
    scanBtn.style.display = "none";
    rtspBox.style.display = "none";
  } else if (type === "demo") {
    uriLabel.innerText = "Synthetic Simulation Mode";
    uriInput.placeholder = "synthetic";
    uriInput.value = "synthetic";
    uriHelper.innerHTML = "Explicit synthetic benchmark demonstration feed.";
    scanBtn.style.display = "none";
    rtspBox.style.display = "none";
  }
}

async function testCameraStream() {
  const btn = document.getElementById("btnTestCamera");
  const vBox = document.getElementById("verificationBox");
  const vBadge = document.getElementById("verificationStatusBadge");
  const vList = document.getElementById("verificationStepsList");

  btn.disabled = true;
  btn.innerText = "⏳ Testing Connection...";
  vBox.style.display = "block";
  vBadge.className = "badge badge-info";
  vBadge.innerText = "CONNECTING & PROBING STREAM...";
  vList.innerHTML = "";

  const payload = {
    source_type: document.getElementById("camSourceType").value,
    source_uri: document.getElementById("camUri").value,
    username: document.getElementById("camUsername") ? document.getElementById("camUsername").value : "",
    password: document.getElementById("camPassword") ? document.getElementById("camPassword").value : ""
  };

  try {
    const res = await fetch("/api/cameras/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const result = await res.json();

    btn.disabled = false;
    btn.innerText = "🧪 Test & Verify Camera Stream";

    const steps = result.steps || [];
    vList.innerHTML = steps.map(s => `
      <div class="verification-step-item">
        <span>${s.status === 'pass' ? '✓' : '✗'} ${s.name}: <small style="color: var(--text-dim);">${s.detail || ''}</small></span>
        <span class="${s.status === 'pass' ? 'step-pass' : 'step-fail'}">${s.status.toUpperCase()}</span>
      </div>
    `).join("");

    if (result.success) {
      vBadge.className = "badge badge-success";
      vBadge.innerText = `✓ ${result.message} (${result.resolution} @ ${result.fps} FPS)`;
    } else {
      vBadge.className = "badge badge-danger";
      vBadge.innerText = `✗ ${result.message}: ${result.error || ''}`;
    }
  } catch (err) {
    btn.disabled = false;
    btn.innerText = "🧪 Test & Verify Camera Stream";
    vBadge.className = "badge badge-danger";
    vBadge.innerText = `Error: ${err.message}`;
  }
}

async function submitAddCamera() {
  const payload = {
    id: document.getElementById("camId").value,
    name: document.getElementById("camName").value,
    source_type: document.getElementById("camSourceType").value,
    source_uri: document.getElementById("camUri").value,
    location: document.getElementById("camLocation").value,
    fps: parseInt(document.getElementById("camFps").value) || 24,
    ai_enabled: document.getElementById("camAiEnabled").checked,
    username: document.getElementById("camUsername") ? document.getElementById("camUsername").value : "",
    password: document.getElementById("camPassword") ? document.getElementById("camPassword").value : ""
  };

  try {
    const res = await fetch("/api/cameras", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.success) {
      closeAddCameraModal();
      window.location.reload();
    } else {
      alert(`Failed to add camera: ${data.detail || 'Unknown error'}`);
    }
  } catch (err) {
    console.error("Error adding camera:", err);
    alert("Error saving camera source.");
  }
}

async function quickConnectWebcam() {
  openAddCameraModal();
  selectSourceTab("webcam");
  document.getElementById("camId").value = "CAM-WEBCAM";
  document.getElementById("camName").value = "Primary Laptop Webcam";
  document.getElementById("camLocation").value = "Local Terminal Station";
  document.getElementById("camUri").value = "0";
  testCameraStream();
}

async function quickScanLocalWebcams() {
  try {
    const res = await fetch("/api/cameras/scan-local", { method: "POST" });
    const data = await res.json();
    if (data.devices && data.devices.length > 0) {
      const dev = data.devices[0];
      document.getElementById("camUri").value = dev.index;
      document.getElementById("camName").value = dev.name;
      alert(`Discovered local camera at index ${dev.index} (${dev.resolution})`);
    } else {
      alert("No local hardware cameras discovered.");
    }
  } catch (e) {
    console.error("Local scan error:", e);
  }
}

async function openScanModal(scanType) {
  const modal = document.getElementById("scanModal");
  if (!modal) return;
  modal.classList.add("active");

  const title = document.getElementById("scanModalTitle");
  const loading = document.getElementById("scanLoading");
  const results = document.getElementById("scanResultsList");

  loading.style.display = "block";
  results.style.display = "none";
  results.innerHTML = "";

  if (scanType === "local") {
    title.innerText = "🔍 Local Hardware Webcams & USB Devices";
    loading.innerText = "Probing local video capture device indices (0..5)...";
    try {
      const res = await fetch("/api/cameras/scan-local", { method: "POST" });
      const data = await res.json();
      loading.style.display = "none";
      results.style.display = "block";

      if (data.devices && data.devices.length > 0) {
        results.innerHTML = data.devices.map(d => `
          <div style="padding: 12px; background: rgba(255,255,255,0.03); border: 1px solid var(--border-color); border-radius: var(--radius-sm); margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
            <div>
              <div style="font-weight: 700;">${d.name}</div>
              <div style="font-size: 11px; color: var(--text-dim);">Index: <code>${d.index}</code> | Res: ${d.resolution} | FPS: ${d.fps}</div>
            </div>
            <button class="btn btn-primary" style="padding: 4px 10px; font-size: 11px;" onclick="useDiscoveredDevice('${d.index}', '${d.name}', 'webcam')">Use Device</button>
          </div>
        `).join("");
      } else {
        results.innerHTML = `<div style="text-align: center; color: var(--text-dim); padding: 20px;">No active local video capture devices detected.</div>`;
      }
    } catch (e) {
      loading.innerText = `Scan error: ${e.message}`;
    }
  } else {
    title.innerText = "🌐 Network ONVIF & IP Cameras";
    loading.innerText = "Broadcasting WS-Discovery probe across local subnet...";
    try {
      const res = await fetch("/api/cameras/scan-network", { method: "POST" });
      const data = await res.json();
      loading.style.display = "none";
      results.style.display = "block";

      if (data.devices && data.devices.length > 0) {
        results.innerHTML = data.devices.map(d => `
          <div style="padding: 12px; background: rgba(255,255,255,0.03); border: 1px solid var(--border-color); border-radius: var(--radius-sm); margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
            <div>
              <div style="font-weight: 700;">${d.name}</div>
              <div style="font-size: 11px; color: var(--text-dim);">IP: ${d.ip} | ${d.suggested_rtsp}</div>
            </div>
            <button class="btn btn-primary" style="padding: 4px 10px; font-size: 11px;" onclick="useDiscoveredDevice('${d.suggested_rtsp}', '${d.name}', 'rtsp')">Use Stream</button>
          </div>
        `).join("");
      } else {
        results.innerHTML = `<div style="text-align: center; color: var(--text-dim); padding: 20px;">No ONVIF IP cameras responded to broadcast. You can add your RTSP URL manually.</div>`;
      }
    } catch (e) {
      loading.innerText = `Network scan error: ${e.message}`;
    }
  }
}

function closeScanModal() {
  const modal = document.getElementById("scanModal");
  if (modal) modal.classList.remove("active");
}

function useDiscoveredDevice(uri, name, type) {
  closeScanModal();
  openAddCameraModal();
  selectSourceTab(type);
  document.getElementById("camUri").value = uri;
  document.getElementById("camName").value = name;
  testCameraStream();
}

// ==================== REAL-TIME PEOPLE INTELLIGENCE POLLER ====================

async function updatePeopleIntelligence() {
  const grid = document.getElementById("peopleIntelligenceGrid");
  if (!grid) return;

  try {
    const res = await fetch("/api/people-intelligence");
    const data = await res.json();
    const subjects = data.subjects || [];

    if (subjects.length === 0) {
      grid.innerHTML = `
        <div style="grid-column: 1/-1; text-align: center; color: var(--text-dim); padding: 18px;">
          No active subjects currently detected in live camera streams.
        </div>
      `;
      return;
    }

    grid.innerHTML = subjects.map(s => {
      const isFall = (s.fall_stage === "CONFIRMED_FALL" || s.current_activity === "falling");
      const isSusp = (s.fall_stage === "POSSIBLE_FALL" || s.fall_stage === "SUSPICIOUS");
      const actBadgeClass = isFall ? 'badge-danger' : isSusp ? 'badge-warning' : 'badge-info';
      const idBadgeClass = s.identity_status === 'recognized' ? 'badge-success' : 'badge-unknown';

      return `
        <div style="background: rgba(255,255,255,0.03); border: 1px solid var(--border-color); border-radius: var(--radius-sm); padding: 12px; display: flex; flex-direction: column; gap: 8px;">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span class="cam-badge">${s.camera_id}</span>
            <span class="badge ${idBadgeClass}">${s.employee_code || 'Unknown Subject'}</span>
          </div>
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-weight: 700; font-size: 13px;">Person #${s.track_id}</span>
            <span class="badge ${actBadgeClass}">${(s.current_activity || 'WORKING').toUpperCase()} (${Math.round((s.activity_confidence || 0.9) * 100)}%)</span>
          </div>
        </div>
      `;
    }).join("");
  } catch (err) {
    console.debug("People intelligence poll note:", err);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initAlertWebSocket();
  updatePeopleIntelligence();
  setInterval(updatePeopleIntelligence, 2000);
});
