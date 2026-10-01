// Real-time CCTV Safety Dashboard Client
let socket = null;
let currentAlertEvent = null;

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
  document.getElementById("modalEventId").innerText = eventData.id || "EVT-ALERT";
  document.getElementById("modalCamera").innerText = eventData.camera_name || eventData.camera_id;
  document.getElementById("modalTrack").innerText = `Track #${eventData.track_id || 'N/A'}`;
  
  const empBadge = document.getElementById("modalEmployee");
  if (eventData.identity_status === "recognized" && eventData.employee_code) {
    empBadge.innerHTML = `<span class="badge badge-success">${eventData.employee_code} (${eventData.employee_name})</span>`;
  } else {
    empBadge.innerHTML = `<span class="badge badge-unknown">Unknown Subject</span>`;
  }

  document.getElementById("modalActivity").innerText = `${(eventData.activity || 'FALL').toUpperCase()} (${Math.round((eventData.activity_confidence || 0.95) * 100)}%)`;
  document.getElementById("modalTimestamp").innerText = eventData.timestamp || new Date().toLocaleTimeString();
  
  const snapshotImg = document.getElementById("modalSnapshot");
  if (eventData.snapshot_path) {
    snapshotImg.src = eventData.snapshot_path;
    snapshotImg.style.display = "block";
  } else {
    snapshotImg.style.display = "none";
  }

  // Open modal
  const modal = document.getElementById("safetyAlertModal");
  if (modal) {
    modal.classList.add("active");
  }

  // Also prepend to table if on events page or dashboard
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
      <button class="btn btn-outline" style="padding: 4px 8px; font-size: 11px;" onclick="openEventDetail('${evt.id}')">View</button>
    </td>
  `;
  tableBody.insertBefore(row, tableBody.firstChild);
}

document.addEventListener("DOMContentLoaded", () => {
  initAlertWebSocket();
});
