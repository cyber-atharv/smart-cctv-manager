# Academic Mini-Project Report
# Smart CCTV AI Safety & Identification Platform
**Fall Detection + Activity Recognition + Person Tracking + Face Enrollment/Recognition + Event Management**

---

## 1. Problem Statement
Workplace slips, trips, and sudden falls represent one of the leading causes of critical workplace trauma and occupational disability across industrial warehouses, eldercare facilities, construction sites, and corporate campuses. Conventional Closed-Circuit Television (CCTV) systems are purely passive recording devices, requiring continuous human monitoring that suffers from fatigue, delayed response times, and an inability to track subjects across obstructed camera zones.

Existing automated surveillance systems often suffer from two major flaws:
1. **Coupled Identity Dependency**: Safety alarm generation is often conflated with facial biometric recognition, causing falls by visitors or un-enrolled personnel to be ignored or delayed.
2. **Instantaneous Frame Sensitivity (High False Positives)**: Simple threshold-based systems trigger false alarms whenever a person bends over, sits quickly, or ties their shoes.

---

## 2. Project Objectives
1. **Real-Time Safety Intelligence**: Implement an automated vision pipeline capable of detecting persons, extracting keypoint postures, and classifying activities (`Standing`, `Sitting`, `Working`, `Falling`) at 15+ FPS.
2. **Decoupled Architecture**: Maintain **Safety Fall Detection** and **Biometric Face Recognition** as strictly independent AI subsystems so that falls by unknown/un-enrolled persons trigger safety alarms identically to recognized employees.
3. **Temporal Fall State Machine**: Implement multi-frame temporal reasoning (`NORMAL ➔ SUSPICIOUS ➔ POSSIBLE_FALL ➔ CONFIRMED_FALL`) to virtually eliminate transient false positives.
4. **Biometric Face Enrollment & Verification**: Build a multi-angle enrollment gallery generating 128-dimensional L2-normalized embeddings for live CCTV identity matching.
5. **Auditable Incident Evidence**: Capture automated pre-event and post-event rolling video buffers alongside high-resolution event snapshots and supervisor review workflows.

---

## 3. Comparison: Existing System vs. Proposed System

| Parameter | Existing Conventional CCTV | Standard Research ML Models | Proposed Smart CCTV AI Platform |
| :--- | :--- | :--- | :--- |
| **Response Type** | Passive (Post-incident review) | Raw frame classification | Real-time active alerts & Dispatch |
| **Temporal Verification** | None (Manual review) | Single-frame prediction | Multi-frame State Machine + Floor Persistence |
| **Tracking** | None | Re-identification only | Multi-Object Tracking with persistent Track IDs |
| **Biometric Decoupling** | Monolithic or Absent | Conflated with classification | Decoupled (Safety operates with Unknown faces) |
| **Incident Evidence** | Continuous raw storage | None | 5s Pre-event + 10s Post-event rolling clips |
| **Supervisor Review** | Manual log inspection | Offline evaluation | Live interactive Confirm/Reject dashboard |

---

## 4. System Architecture

```
CCTV / Webcam / Video File / Synthetic Stream
                     │
                     ▼
             [ Video Gateway ]
                     │
                     ▼
         [ Frame Buffer / Scheduler ]
                     │
                     ▼
     [ Person Detection (YOLOv8 / HOG) ]
                     │
                     ▼
   [ Multi-Object Tracking (Persistent IDs) ]
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
 [ Pose & Keypoints ]    [ Upper-body ROI ]
   (MediaPipe Pose)             │
          │                     ▼
          ▼              [ Face Detection ]
 [ Feature Extractor ]          │
   (16 Kinematic Feats)         ▼
          │              [ Face Embedder ]
          ▼              (128-d L2 Norm)
 [ ML Classifier ]              │
 (Random Forest v1.0)           ▼
          │             [ Employee Gallery ]
          ▼                     │
 [ Temporal Verifier ]          ▼
(NORMAL➔SUSP➔FALL)      [ Recognized / Unknown ]
          │                     │
          └──────────┬──────────┘
                     ▼
        [ Event Correlation Engine ]
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   [ SQLite DB ] [ MP4 Clip ] [ WebSocket ]
        │            │            │
        └────────────┼────────────┘
                     ▼
      [ CCTV Supervisor Web Dashboard ]
```

---

## 5. Dataset & Feature Engineering

### 5.1 Dataset Schema & Leakage Prevention
To prevent data leakage, training and test datasets were split at the **Subject Group Level** (`GroupKFold` / `subject_id` split) rather than random frame sampling. This guarantees that nearly identical frames from the same individual/recording never appear across both training and testing partitions.

### 5.2 Standardized 16-Feature Kinematic Vector
| # | Feature Name | Mathematical Description / Biomechanical Role |
| :-: | :--- | :--- |
| 1 | `body_angle` | Torso vector inclination from vertical axis ($\theta = \arctan(\Delta x / \Delta y)$). Upright $\approx 0^\circ$, Fallen $\approx 70^\circ-90^\circ$. |
| 2 | `shoulder_angle` | Angle of shoulder vector relative to horizontal plane. |
| 3 | `hip_angle` | Angle of hip vector relative to horizontal plane. |
| 4 | `knee_angle_left` | Left knee flexion angle computed at the knee vertex ($\angle(\text{hip}, \text{knee}, \text{ankle})$). |
| 5 | `knee_angle_right` | Right knee flexion angle. |
| 6 | `head_angle` | Angle from mid-shoulder to nose vertex. |
| 7 | `body_height` | Normalized vertical span of body bounding box ($\Delta y_{\text{max}} - \Delta y_{\text{min}}$). |
| 8 | `body_width` | Normalized horizontal span of body bounding box ($\Delta x_{\text{max}} - \Delta x_{\text{min}}$). |
| 9 | `height_width_ratio` | $\text{Aspect Ratio} = \text{height} / (\text{width} + \epsilon)$. Drops from $>2.0$ to $<0.8$ during a fall. |
| 10 | `vertical_velocity` | First-order difference of mid-hip vertical position ($\Delta y_{\text{hip}} / \Delta t$). |
| 11 | `horizontal_velocity`| First-order difference of mid-hip horizontal position ($\Delta x_{\text{hip}} / \Delta t$). |
| 12 | `movement_speed` | Euclidean magnitude of centroid displacement ($\sqrt{v_x^2 + v_y^2}$). |
| 13 | `bbox_area` | Bounding box spatial footprint ratio. |
| 14 | `landmark_visibility`| Mean joint confidence score. |
| 15 | `hip_center_y` | Normalized vertical coordinate of hip in frame (floor proximity). |
| 16 | `head_to_ankle_y_diff`| Vertical span from head to ankles. |

---

## 6. Machine Learning Models & Experimental Results

### 6.1 Frame-Level Evaluation on Held-Out Test Set (1,200 Samples across 6 Strictly Unseen Subjects)
> [!IMPORTANT]
> **Academic Note on Evaluation Scope**: The metrics reported below reflect **frame-level classifier test accuracy** on the held-out subject test partition. They measure the classifier's ability to distinguish isolated 16-feature kinematic vectors across unseen subjects and must not be described as "real-world fall-detection accuracy." Real-world system performance is governed by end-to-end multi-frame temporal dynamics, floor persistence verification, and environmental factors.

- **Dataset Configuration**: 6,000 samples across 30 distinct subjects, incorporating hard negative postures (crouching to tie shoes, bending over to pick up objects, and rapid sitting).
- **Split Strategy**: Group-Aware Subject Partitioning (24 subjects Train / 6 subjects Test) to strictly prevent identical-frame data leakage.
- **Decision Tree Classifier** (`max_depth=6`): **98.00% Frame-Level Test Accuracy**
- **Random Forest Classifier** (`n_estimators=100`, `max_depth=10`): **99.42% Frame-Level Test Accuracy**

### 6.2 Classification Report (Random Forest Model)
```
              precision    recall  f1-score   support

     falling       0.98      1.00      0.99       176
     sitting       1.00      1.00      1.00       314
    standing       1.00      1.00      1.00       296
     working       1.00      0.99      0.99       414

    accuracy                           0.99      1200
   macro avg       0.99      1.00      0.99      1200
weighted avg       0.99      0.99      0.99      1200
```

### 6.3 Multi-Class Confusion Matrix (1,200 Test Frames)
```
Classes: ['falling', 'sitting', 'standing', 'working']
[[176   0   0   0]   <- Ground Truth: Falling (176 frames evaluated)
 [  0 313   0   1]   <- Ground Truth: Sitting
 [  0   0 295   1]   <- Ground Truth: Standing
 [  4   1   0 409]]  <- Ground Truth: Working (includes hard negative crouch/bend)
```

### 6.4 False Positive & False Negative Analysis on Test Partition
- **Test-Partition True Positives (TP)**: 176 falling frames correctly classified ($176/176$).
- **Test-Partition False Negatives (FN)**: 0 false negatives observed specifically within the evaluated 176 test frames (does not imply zero false negatives across arbitrary real-world fall variations).
- **Raw Instantaneous Frame False Positives (FP)**: 4 working/bending frames classified as `falling` at the raw instantaneous single-frame level.
- **Role of the Temporal State Machine in Real-World Deployment**: 
  These 4 single-frame false positives demonstrate why raw frame classifiers cannot directly drive industrial alarms. The **Temporal Fall Verifier** (`floor_persistence_frames = 12`) filters these transient candidate frames: when a subject bends over to retrieve an object or tie shoes and quickly recovers, the state machine resets from `SUSPICIOUS` back to `NORMAL` before an alert is dispatched.
- **Test-Partition Fall Recall (Sensitivity)**: $100.00\%$ ($176/176$ test frames)
- **Test-Partition Fall Precision**: $97.78\%$
- **Test-Partition Fall Specificity**: $99.61\%$

---

## 7. Temporal Fall Verification State Machine
Single-frame classifiers are inherently susceptible to transient motion noise (e.g. quick shoe tying or picking up tools). The **Temporal State Machine** enforces:
1. **`NORMAL`**: Person posture and movement conform to standard upright activity.
2. **`SUSPICIOUS`**: Abnormal downward velocity or tilted posture detected for $\ge 3$ consecutive frames.
3. **`POSSIBLE_FALL`**: Posture remains abnormal for $\ge 6$ consecutive frames.
4. **`CONFIRMED_FALL`**: Subject remains horizontal/near floor for $\ge 12$ floor persistence frames without recovery motion.
5. **Debounce Control**: 8-second cooldown to avoid duplicate incident dispatches for the same event.

---

## 8. Biometric Face Enrollment & Decoupling Rule
1. **Enrollment Workflow**: Captures multi-angle samples (Front, Left 30°, Right 30°), validates sharpness and single-face presence, extracts 128-d L2-normalized embeddings, and stores vectors into SQLite.
2. **Safety Rule**: If a face is obscured or unregistered, `identity_status = "unknown"`. Safety fall detection executes unconditionally.

---

## 9. Limitations & Future Scope
- **Current Limitations**: Severe body occlusions (e.g. fallen behind large heavy machinery) may reduce keypoint visibility.
- **Future Scope**:
  - Multi-camera cross-view track handoff.
  - Thermal infrared camera stream fusion for dark environments.
  - Direct edge deployment on NVIDIA Jetson / Intel OpenVINO.

---

## 10. Conclusion
The Smart CCTV AI Safety & Identification Platform successfully demonstrates an academic, production-ready computer vision architecture. By combining geometric pose feature extraction, group-aware ML classification, temporal state verification, and decoupled biometric identification, the platform achieves robust real-time incident alerting while adhering to strict privacy and engineering best practices.
