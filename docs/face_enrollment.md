# Face Enrollment & Biometric Identity Architecture

## 1. Biometric Safety Decoupling Rule
Biometric facial identification is an **optional identity enrichment layer** and is strictly decoupled from the core **Safety Intelligence Subsystem**.
- An un-enrolled visitor falling produces: `employee_id = null`, `identity_status = "unknown"`, `activity = "falling"`.
- A registered employee falling produces: `employee_id = 1`, `employee_code = "EMP001"`, `activity = "falling"`.
Both cases dispatch identical high-priority safety incidents, audio-visual alarms, snapshots, and video clips.

## 2. Enrollment Workflow
1. **Employee Profile Creation**: Register `employee_code`, `name`, `department`.
2. **Multi-Angle Calibration**: Capture 3 guided samples:
   - **Front Angle** (`sample_index = 0`)
   - **Left Profile ~30°** (`sample_index = 1`)
   - **Right Profile ~30°** (`sample_index = 2`)
3. **Quality Validation**:
   - Single face verification (multiple faces or zero faces rejected).
   - Blur/sharpness detection via Laplacian variance ($\sigma^2 \ge 30.0$).
   - Minimum face bounding box size ($\ge 40\text{px}$).
4. **Embedding Extraction**: Lightweight convolutional feature extractor generates a 128-dimensional unit vector normalized by L2 norm ($||\mathbf{v}||_2 = 1.0$).
5. **Database Storage**: Vectors stored as JSON float arrays in `face_profiles` table. Raw biometric photos are not permanently retained.

## 3. Live CCTV Matching
1. The upper region of tracked person bounding boxes is cropped.
2. If a face is found, its 128-d embedding is extracted.
3. Cosine similarity is computed against all enrolled vectors in the gallery cache:
   $$\text{Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{||\mathbf{u}||_2 ||\mathbf{v}||_2}$$
4. If $\text{Similarity} \ge 0.55$, the persistent track is assigned the employee identity.
