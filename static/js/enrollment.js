// Biometric Face Enrollment Client Script
let currentStream = null;
let currentPoseAngle = "front";
let currentSampleIndex = 0;

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
  const employeeSelect = document.getElementById("enrollEmployeeSelect");
  const employeeId = employeeSelect.value;
  if (!employeeId) {
    alert("Please select an employee first.");
    return;
  }

  const video = document.getElementById("enrollmentVideo");
  const fileInput = document.getElementById("enrollmentFileInput");
  const statusBox = document.getElementById("enrollmentStatusMsg");

  let blob = null;

  if (video.srcObject && video.style.display !== "none") {
    // Capture frame from video canvas
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
  formData.append("employee_id", employeeId);
  formData.append("pose_angle", currentPoseAngle);
  formData.append("sample_index", currentSampleIndex);
  formData.append("image", blob, `sample_${currentPoseAngle}.jpg`);

  statusBox.innerText = `Processing and extracting 128-d biometric embedding for ${currentPoseAngle} angle...`;
  statusBox.className = "alert-detail-item";

  try {
    const res = await fetch("/api/face/enroll", {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    
    if (res.ok && data.success) {
      statusBox.innerHTML = `<span style="color: var(--success); font-weight: bold;">✔ Enrolled ${currentPoseAngle.toUpperCase()} profile successfully! Total samples: ${data.total_samples}</span>`;
      
      // Mark step as completed
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
