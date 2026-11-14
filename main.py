
import cv2
import time
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from camera import Camera
from pose_analyser import PoseAnalyzer
from pose_evaluator import PoseEvaluator
from predict_pose import predict_pose


# -------------------------------
# 1. Model path
# -------------------------------

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "pose_landmarker_full.task"


# -------------------------------
# 2. Initialize MediaPipe
# -------------------------------

base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

landmarker = vision.PoseLandmarker.create_from_options(
    options
)


# -------------------------------
# 3. Create class objects
# -------------------------------

camera = Camera()

analyzer = PoseAnalyzer()

evaluator = PoseEvaluator()


# -------------------------------
# 4. Camera loop
# -------------------------------

try:

    while True:

        ret, frame = camera.read_frame()

        if not ret:
            break

        height, width = frame.shape[:2]

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Timestamp for VIDEO mode
        timestamp_ms = time.monotonic_ns() // 1_000_000

        # Detect landmarks
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # -------------------------------
        # 5. Process detected pose
        # -------------------------------

        if result.pose_landmarks:

            landmarks = result.pose_landmarks[0]

            # Store landmarks
            analyzer.set_landmarks(landmarks)

            # Draw circles and skeleton
            analyzer.draw_skeleton(frame)

            # Get all joint angles
            angles = analyzer.get_all_angles(
                width,
                height
            )

            # Predict yoga pose
            pose_name, confidence = predict_pose(
                landmarks,
                width,
                height
            )

            # Display predicted pose
            if pose_name is not None:

                cv2.putText(
                    frame,
                    f"Pose: {pose_name}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Confidence: {confidence * 100:.1f}%",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

                # Evaluate all joint angles
                evaluation = evaluator.evaluate_pose(
                    pose_name,
                    angles
                )

                # Display joint angles
                y_position = 95

                for joint_name, angle in angles.items():

                    if angle is None:
                        text = f"{joint_name}: Not visible"

                    else:
                        status = evaluation.get(
                            joint_name,
                            "Not evaluated"
                        )

                        text = (
                            f"{joint_name}: "
                            f"{angle:.1f} deg - {status}"
                        )

                    cv2.putText(
                        frame,
                        text,
                        (10, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (255, 255, 255),
                        1
                    )

                    y_position += 22

                # Display feedback
                feedback = evaluator.get_feedback(
                    evaluation
                )

                y_position += 10

                for message in feedback:

                    cv2.putText(
                        frame,
                        message,
                        (10, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        1
                    )

                    y_position += 22

            else:

                cv2.putText(
                    frame,
                    "Unable to predict pose",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

        else:

            cv2.putText(
                frame,
                "No pose detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # -------------------------------
        # 6. Display FPS
        # -------------------------------

        cv2.putText(
            frame,
            f"FPS: {camera.fps:.1f}",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        # -------------------------------
        # 7. Display camera
        # -------------------------------

        cv2.imshow(
            "Yoga Wellness - Pose Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


finally:

    camera.release()
    landmarker.close()
