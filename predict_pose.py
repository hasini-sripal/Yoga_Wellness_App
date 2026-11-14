from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "saved_models"

clf = joblib.load(MODEL_DIR / "asana_classifier.joblib")

# Must match prepare_data.py exactly
JOINTS = [
    ("l_elbow", 11, 13, 15), ("r_elbow", 12, 14, 16),
    ("l_shoulder", 13, 11, 23), ("r_shoulder", 14, 12, 24),
    ("l_hip", 11, 23, 25), ("r_hip", 12, 24, 26),
    ("l_knee", 23, 25, 27), ("r_knee", 24, 26, 28),
]


def _angle(a, b, c):
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def predict_pose(landmarks, width, height, min_confidence=0.6):
    """
    landmarks : result.pose_landmarks[0] from MediaPipe PoseLandmarker
    width, height : size of the frame in pixels
    returns : (pose_name, confidence). pose_name is None if confidence is too low.
    """
    coords = np.array(
        [[lm.x * width, lm.y * height, lm.z * width] for lm in landmarks],
        dtype=np.float32,
    )

    angles = [_angle(coords[a], coords[b], coords[c]) for _, a, b, c in JOINTS]

    hip_center = (coords[23] + coords[24]) / 2
    coords -= hip_center
    torso = np.linalg.norm((coords[11] + coords[12]) / 2)
    if torso < 1e-6:
        return None, 0.0
    coords /= torso

    features = np.concatenate([coords.flatten(), angles])
    X = pd.DataFrame([features], columns=clf.feature_names_in_)

    probs = clf.predict_proba(X)[0]
    best = int(probs.argmax())
    confidence = float(probs[best])
    if confidence < min_confidence:
        return None, confidence
    return str(clf.classes_[best]), confidence