// Biometric Face Enrollment Client Script
let currentStream = null;
let currentPoseAngle = "front";
let currentSampleIndex = 0;
let currentEnrollMode = "existing";

function switchEnrollMode(mode) {
  currentEnrollMode = mode;
  const existingSection = document.getElementById("existingPersonSection");
  const newSection = document.getElementById("newPersonSection");
  const selectBtn = document.getElementById("modeSelectBtn");
  const newBtn = document.getElementById("modeNewBtn");

  if (mode === "existing") {
    existingSection.style.display = "block";
    newSection.style.display = "none";
    selectBtn.className = "btn btn-primary";
    newBtn.className = "btn btn-outline";
  } else {
    existingSection.style.display = "none";
    newSection.style.display = "flex";
    selectBtn.className = "btn btn-outline";
    newBtn.className = "btn btn-primary";
    const nameInput = document.getElementById("newEmpName");
    if (nameInput) nameInput.focus();
  }
}

async function startEnrollmentCamera() {
  const video = document.getElementById("enrollmentVideo");
  const placeholder = document.getElementById("cameraPlaceholder");
  
  try {
    currentStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = currentStream;
    video.style.display = "block";
    if (placeholder) placeholder.style.display = "none";
  } catch (err) {
    console.warn("Webcam access error or denied:", err);
    alert("Camera unavailable or permission denied. You can also use the image upload option below.");
  }
}

function selectPoseStep(angle, index) {
  currentPoseAngle = angle;
  currentSampleIndex = index;
  
  document.querySelectorAll(".pose-step").forEach((el, idx) => {
    if (idx === index) {
      el.classList.add("active");
    } else {
      el.classList.remove("active");
    }
  });
}

async function captureAndEnrollSample() {
  const statusBox = document.getElementById("enrollmentStatusMsg");
  const video = document.getElementById("enrollmentVideo");
  const fileInput = document.getElementById("enrollmentFileInput");

  let blob = null;

  if (video.srcObject && video.style.display !== "none") {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.95));
  } else if (fileInput.files && fileInput.files[0]) {
    blob = fileInput.files[0];
  } else {
    alert("Please start the camera or select an image file to enroll.");
    return;
  }

  const formData = new FormData();
  if (currentEnrollMode === "existing") {
    const employeeSelect = document.getElementById("enrollEmployeeSelect");
    const employeeId = employeeSelect ? employeeSelect.value : "";
    if (!employeeId) {
      alert("Please select an employee or switch to 'Register New Person'.");
      return;
    }
    formData.append("employee_id", employeeId);
  } else {
    const name = document.getElementById("newEmpName").value.trim();
    if (!name) {
      alert("Please enter the person's Full Name.");
      return;
    }
    const code = document.getElementById("newEmpCode").value.trim() || `EMP-${Date.now().toString().slice(-4)}`;
    const dept = document.getElementById("newEmpDept").value.trim() || "Operations";
    formData.append("name", name);
    formData.append("employee_code", code);
    formData.append("department", dept);
  }

  formData.append("pose_angle", currentPoseAngle);
  formData.append("sample_index", currentSampleIndex);
  formData.append("file", blob, `sample_${currentPoseAngle}.jpg`);
  formData.append("image", blob, `sample_${currentPoseAngle}.jpg`);

  statusBox.innerText = `Extracting 128-d biometric embedding for ${currentPoseAngle} angle...`;
  statusBox.className = "alert-detail-item";

  try {
    const res = await fetch("/api/face/enroll", {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    
    if (res.ok && data.success) {
      const empLabel = data.employee_code ? `Personnel ${data.employee_code}` : "Profile";
      statusBox.innerHTML = `<span style="color: var(--success); font-weight: bold;">✔ Enrolled ${empLabel} (${currentPoseAngle.toUpperCase()}) successfully! Total samples: ${data.total_samples}</span>`;
      
      const activeStep = document.querySelector(`.pose-step.active`);
      if (activeStep) activeStep.classList.add("completed");

      // Auto advance to next angle
      if (currentSampleIndex < 2) {
        const nextIdx = currentSampleIndex + 1;
        const nextAngle = nextIdx === 1 ? "left" : "right";
        setTimeout(() => selectPoseStep(nextAngle, nextIdx), 800);
      }
    } else {
      statusBox.innerHTML = `<span style="color: var(--danger); font-weight: bold;">✖ Enrollment Error: ${data.detail || data.error}</span>`;
    }
  } catch (err) {
    console.error("Enrollment error:", err);
    statusBox.innerHTML = `<span style="color: var(--danger);">Network or processing error during enrollment.</span>`;
  }
}
