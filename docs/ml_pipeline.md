# Machine Learning Pipeline Documentation

## 1. Feature Representation
Every detected human posture is transformed into a standardized 16-dimensional vector:
- `body_angle`: Torso vector angle relative to vertical.
- `shoulder_angle`, `hip_angle`: Horizontal axis alignment angles.
- `knee_angle_left`, `knee_angle_right`: Knee joint flexion angles ($\angle(\text{hip}, \text{knee}, \text{ankle})$).
- `head_angle`: Deviation of head from shoulder midpoint.
- `body_height`, `body_width`: Bounding box dimensions.
- `height_width_ratio`: Body aspect ratio ($\text{height} / \text{width}$).
- `vertical_velocity`, `horizontal_velocity`, `movement_speed`: Kinematic displacement vectors.
- `bbox_area`, `landmark_visibility`, `hip_center_y`, `head_to_ankle_y_diff`.

## 2. Prevention of Data Leakage
Standard random train/test splits cause severe over-optimistic test leakage when multiple frames from the same video or subject appear in both sets.
We employ **Group-Aware Subject Splitting** (`GroupKFold` across `subject_id`s), ensuring that model generalizability is measured exclusively on unseen subjects.

## 3. Training & Evaluation
- **Champion Model**: `RandomForestClassifier` (`n_estimators=100`, `max_depth=8`)
- **Baseline Model**: `DecisionTreeClassifier` (`max_depth=6`)
- **Classes**: `standing`, `sitting`, `working`, `falling`

## 4. Artifact Versioning
- Weights: `models/fall_detection_model.pkl`
- Feature Schema: `models/feature_columns.json`
- Model Metadata: `models/metadata.json`
