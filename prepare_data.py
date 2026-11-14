import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tqdm import tqdm

from features import FEATURE_COLUMNS, landmarks_to_features

# ----------------------------- CONFIG -----------------------------
ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = ROOT / "dataset"
OUTPUT_PATH = ROOT / "landmarks.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "pose_landmarker_full.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")

KEEP = None            # e.g. {"tree", "plank"} to use only some pose folders
EXTS = (".jpg", ".jpeg", ".png", ".bmp")
ADD_MIRRORED = True    # also add a flipped copy of each image (the camera feed is mirrored)
# ------------------------------------------------------------------

MODEL_DIR.mkdir(exist_ok=True)
if not MODEL_PATH.exists():
    print("Downloading pose model...")
    urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))


def features_from_image(image_bgr, landmarker):
    h, w = image_bgr.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.pose_landmarks:
        return None
    return landmarks_to_features(result.pose_landmarks[0], w, h)


def find_splits(root):
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    splits = [(s, root / s) for s in ("train", "val", "test") if (root / s).is_dir()]
    return splits if splits else [("all", root)]


def main():
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )

    splits = find_splits(DATASET_ROOT)
    print("Splits found:", [s for s, _ in splits])

    rows, skipped = [], {}

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for split, split_dir in splits:
            classes = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
            if KEEP is not None:
                classes = [c for c in classes if c in KEEP]
            print(f"[{split}] classes: {classes}")

            for label in classes:
                folder = split_dir / label
                files = sorted(f.name for f in folder.iterdir()
                               if f.name.lower().endswith(EXTS))
                key = (split, label)
                skipped[key] = {"unreadable": 0, "no_pose": 0}

                for fn in tqdm(files, desc=f"{split}/{label}"):
                    img = cv2.imread(str(folder / fn))
                    if img is None:
                        skipped[key]["unreadable"] += 1
                        continue

                    versions = [(img, 0)]
                    if ADD_MIRRORED:
                        versions.append((cv2.flip(img, 1), 1))

                    for version, flipped in versions:
                        feats = features_from_image(version, landmarker)
                        if feats is None:
                            skipped[key]["no_pose"] += 1
                            continue
                        rows.append(list(feats) + [label, fn, split, flipped])

    if not rows:
        raise ValueError("No poses detected. Check the dataset folder and your images.")

    df = pd.DataFrame(
        rows, columns=FEATURE_COLUMNS + ["label", "filename", "split", "flipped"])
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nSkipped (split, class): counts")
    for key, val in skipped.items():
        print(" ", key, val)
    print("\nSamples per split/class:")
    print(df.groupby(["split", "label"]).size())
    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()