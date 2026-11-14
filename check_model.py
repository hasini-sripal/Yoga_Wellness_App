import sys
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from predict_pose import predict_pose

ROOT = Path(__file__).resolve().parent.parent
EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Folder with one subfolder per pose. Default: dataset/test
folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dataset" / "test"

options = vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(
        model_asset_path=str(ROOT / "models" / "pose_landmarker_full.task")),
    running_mode=vision.RunningMode.IMAGE,
)

total = correct = 0
with vision.PoseLandmarker.create_from_options(options) as landmarker:
    for pose_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        n = ok = 0
        wrong = {}
        for img_path in pose_dir.iterdir():
            if img_path.suffix.lower() not in EXTS:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not result.pose_landmarks:
                continue
            name, conf = predict_pose(result.pose_landmarks[0], w, h)
            n += 1
            if name == pose_dir.name:
                ok += 1
            else:
                wrong[name] = wrong.get(name, 0) + 1
        total += n
        correct += ok
        pct = 100 * ok / n if n else 0
        print(f"{pose_dir.name:<10} {ok}/{n} correct ({pct:.0f}%)  mistaken for: {wrong}")

print(f"\nOverall: {correct}/{total} ({100 * correct / max(total, 1):.0f}%)")