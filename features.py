import numpy as np

# Same names and landmark indices as PoseAnalyzer in pose_analyser.py.
# The middle index is the joint being measured.
JOINTS = {
    "left_elbow": (11, 13, 15),
    "right_elbow": (12, 14, 16),
    "left_shoulder": (13, 11, 23),
    "right_shoulder": (14, 12, 24),
    "left_hip": (11, 23, 25),
    "right_hip": (12, 24, 26),
    "left_knee": (23, 25, 27),
    "right_knee": (24, 26, 28),
    "left_ankle": (25, 27, 31),
    "right_ankle": (26, 28, 32),
}

FEATURE_COLUMNS = (
    [f"xyz_{i}" for i in range(33 * 3)]
    + [f"angle_{name}" for name in JOINTS]
)


def angle(a, b, c):
    """Angle at point b (degrees) formed by a-b-c."""
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def landmarks_to_features(landmarks, width, height):
    """
    landmarks : the 33 MediaPipe landmarks of one person
    returns   : 109 numbers (99 body coordinates + 10 joint angles), or None if unusable
    """
    # x, y in pixels so angles are not distorted by the image's aspect ratio
    coords = np.array(
        [[lm.x * width, lm.y * height, lm.z * width] for lm in landmarks],
        dtype=np.float32,
    )

    # Angles first: they don't change when the body is shifted or rescaled
    angles = [angle(coords[a], coords[b], coords[c]) for a, b, c in JOINTS.values()]

    # Centre on the hips, scale by torso length
    hip_center = (coords[23] + coords[24]) / 2
    coords -= hip_center
    torso = np.linalg.norm((coords[11] + coords[12]) / 2)
    if torso < 1e-6:
        return None
    coords /= torso

    return np.concatenate([coords.flatten(), angles])