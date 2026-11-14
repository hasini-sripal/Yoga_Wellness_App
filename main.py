
import cv2
import time

from camera import Camera
from pose_analyser import PoseAnalyzer
from pose_evaluator import PoseEvaluator

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -------------------------------
# 1. Initialize MediaPipe
# -------------------------------

base_options = python.BaseOptions(
    model_asset_path="C:\\Users\\harsh\\OneDrive\\Desktop\\yoga_pose_detection\\models\\pose_landmarker_full.task"
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
# 2. Create class objects
# -------------------------------

camera = Camera()

analyzer = PoseAnalyzer()

evaluator = PoseEvaluator()


# -------------------------------
# 3. Start camera loop
# -------------------------------

try:

    while True:

        ret, frame = camera.read_frame()

        if not ret:
            break

        # Convert OpenCV BGR image to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Timestamp in milliseconds
        timestamp_ms = time.monotonic_ns() // 1_000_000

        # Detect pose
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # -------------------------------
        # 4. Analyze detected landmarks
        # -------------------------------

        if result.pose_landmarks:

            landmarks = result.pose_landmarks[0]

            analyzer.set_landmarks(landmarks)

            # Draw skeleton and landmark locations
            frame = analyzer.draw_skeleton(frame)

            # Calculate every joint angle
            angles = analyzer.get_all_angles()

            # Evaluate all calculated angles
            evaluation = evaluator.evaluate_pose(angles)

            # Display angle and evaluation information
            y_position = 30

            for joint_name, angle in angles.items():

                if angle is None:
                    text = f"{joint_name}: Not visible"
                else:
                    status = evaluation[joint_name]
                    text = f"{joint_name}: {angle:.1f} deg - {status}"

                cv2.putText(
                    frame,
                    text,
                    (10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1
                )

                y_position += 20

        else:

            cv2.putText(
                frame,
                "No pose detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # -------------------------------
        # 5. Display FPS
        # -------------------------------

        cv2.putText(
            frame,
            f"FPS: {camera.fps:.1f}",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # Display camera
        cv2.imshow("Yoga Wellness - Pose Detection", frame)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


finally:

    camera.release()
    landmarker.close()
