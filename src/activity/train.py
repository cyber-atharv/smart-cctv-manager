import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import logging
from src.features.feature_extractor import FEATURE_COLUMNS
from src.data.dataset_adapters import BiomechanicalRealisticGenerator

logger = logging.getLogger("cctv.ml_training")

def train_and_export_models(data_dir: Path, models_dir: Path):
    """
    Executes the comprehensive ML training workflow:
    1. Realistic Multi-Subject Data Generation (incorporating Hard Negatives like shoe-tying, crouching, rapid sitting)
    2. Group-Aware Subject Split (Preventing Data Leakage)
    3. Model Training (Decision Tree vs Random Forest)
    4. Evaluation & Metrics Calculation (Honest, realistic non-idealized reporting)
    5. Artifact Export: model.pkl, feature_columns.json, metadata.json
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Realistic Multi-Subject Dataset Preparation ===")
    generator = BiomechanicalRealisticGenerator(n_subjects=30, samples_per_subject=200, random_state=42)
    df = generator.generate()
    
    raw_path = data_dir / "raw" / "fall_activity_dataset.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)
    print(f"Dataset generated: {len(df)} samples across {df['subject_id'].nunique()} subjects (Saved to: {raw_path})")

    print("\n=== Step 2: Group-Aware Train/Test Split (Preventing Leakage) ===")
    unique_subjects = list(df["subject_id"].unique())
    # 24 subjects (80%) for training, 6 subjects (20%) strictly held-out for test
    train_subs, test_subs = train_test_split(unique_subjects, test_size=0.20, random_state=42)
    
    train_df = df[df["subject_id"].isin(train_subs)].copy()
    test_df = df[df["subject_id"].isin(test_subs)].copy()

    X_train = np.array(train_df[FEATURE_COLUMNS], dtype=np.float32)
    y_train = np.array(train_df["label"])
    X_test = np.array(test_df[FEATURE_COLUMNS], dtype=np.float32)
    y_test = np.array(test_df["label"])

    print(f"Train set: {len(X_train)} samples across {len(train_subs)} subjects: {train_subs[:4]}...")
    print(f"Test set:  {len(X_test)} samples across {len(test_subs)} subjects: {test_subs}")

    print("\n=== Step 3: Model Training (Decision Tree & Random Forest) ===")
    # 1. Decision Tree (Educational Baseline)
    dt_model = DecisionTreeClassifier(max_depth=6, random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    dt_acc = accuracy_score(y_test, y_pred_dt)
    print(f"Decision Tree Test Accuracy: {dt_acc * 100:.2f}%")

    # 2. Random Forest (Production Champion Model)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Test Accuracy: {rf_acc * 100:.2f}%")

    print("\n=== Step 4: Multi-Class Evaluation & Confusion Matrix ===")
    classes = sorted(list(np.unique(y_test)))
    report_dict = classification_report(y_test, y_pred_rf, output_dict=True)
    report_text = classification_report(y_test, y_pred_rf)
    print(report_text)

    cm = confusion_matrix(y_test, y_pred_rf, labels=classes)
    print("Confusion Matrix:")
    print(f"Classes: {classes}")
    print(cm)

    # Fall-specific diagnostic metrics
    fall_idx = classes.index("falling")
    tp_fall = int(cm[fall_idx, fall_idx])
    fn_fall = int(np.sum(cm[fall_idx, :]) - tp_fall)
    fp_fall = int(np.sum(cm[:, fall_idx]) - tp_fall)
    tn_fall = int(np.sum(cm) - (tp_fall + fn_fall + fp_fall))

    fall_recall = tp_fall / max(1, (tp_fall + fn_fall))
    fall_prec = tp_fall / max(1, (tp_fall + fp_fall))
    fall_spec = tn_fall / max(1, (tn_fall + fp_fall))

    print(f"\nFall Detection Analysis on Unseen Subjects:")
    print(f"  True Positives (TP):  {tp_fall}")
    print(f"  False Positives (FP): {fp_fall} (Raw single-frame level; filtered by Temporal Verifier)")
    print(f"  False Negatives (FN): {fn_fall}")
    print(f"  Fall Sensitivity (Recall): {fall_recall * 100:.2f}%")
    print(f"  Fall Precision:           {fall_prec * 100:.2f}%")
    print(f"  Fall Specificity:         {fall_spec * 100:.2f}%")

    print("\n=== Step 5: Exporting Versioned Artifacts ===")
    # 1. Model Pickle
    model_export_path = models_dir / "fall_detection_model.pkl"
    with open(model_export_path, "wb") as f:
        pickle.dump(rf_model, f)
    print(f"Exported Model: {model_export_path}")

    # 2. Feature Columns JSON
    features_export_path = models_dir / "feature_columns.json"
    with open(features_export_path, "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)
    print(f"Exported Feature Schema: {features_export_path}")

    # 3. Model Metadata JSON
    metadata = {
        "model_name": "fall_detection_random_forest",
        "version": "1.1-realistic",
        "dataset_type": "Biomechanical Realistic Multi-Subject Benchmark with Hard Negatives",
        "n_subjects": len(unique_subjects),
        "split_method": "Group-Aware Subject Split (24 Train / 6 Test)",
        "algorithm": "RandomForestClassifier",
        "parameters": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
        "classes": classes,
        "features": FEATURE_COLUMNS,
        "metrics": {
            "test_accuracy": round(rf_acc, 4),
            "decision_tree_accuracy": round(dt_acc, 4),
            "per_class_f1": {cls: round(report_dict[cls]["f1-score"], 4) for cls in classes},
            "fall_recall": round(fall_recall, 4),
            "fall_precision": round(fall_prec, 4),
            "fall_specificity": round(fall_spec, 4),
            "raw_frame_false_positives": fp_fall,
            "raw_frame_false_negatives": fn_fall
        },
        "feature_importances": {
            feat: round(float(imp), 4)
            for feat, imp in zip(FEATURE_COLUMNS, rf_model.feature_importances_)
        }
    }
    metadata_export_path = models_dir / "metadata.json"
    with open(metadata_export_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Exported Metadata: {metadata_export_path}")

    return metadata

if __name__ == "__main__":
    from src.config import config
    train_and_export_models(config.data_dir, config.models_dir)
