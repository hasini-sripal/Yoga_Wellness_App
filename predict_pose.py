
from pathlib import Path

import joblib

from features import landmarks_to_features


# Project root: assumes this file is inside a src folder
ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "asana_classifier.joblib"

# Load trained classifier once when this module is imported
model = joblib.load(MODEL_PATH)


def predict_pose(landmarks, width, height):

    features = landmarks_to_features(
        landmarks,
        width,
        height
    )

    if features is None:
        return None, 0.0

    # Convert features into the format expected by sklearn
    features = features.reshape(1, -1)

    probabilities = model.predict_proba(features)[0]

    best_index = probabilities.argmax()

    pose_name = model.classes_[best_index]
    confidence = float(probabilities[best_index])

    return pose_name, confidence
