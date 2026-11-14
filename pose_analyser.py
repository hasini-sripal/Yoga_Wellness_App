
import cv2
import numpy as np


class PoseAnalyzer:

    def __init__(self):

        self.landmarks = None
        self.coordinates = None
        self.visibility = None

        # Three landmark indices define each angle.
        # The middle index is the joint being measured.

        self.joints = {
            "left_elbow": (11, 13, 15),
            "right_elbow": (12, 14, 16),

            "left_shoulder": (13, 11, 23),
            "right_shoulder": (14, 12, 24),

            "left_hip": (11, 23, 25),
            "right_hip": (12, 24, 26),

            "left_knee": (23, 25, 27),
            "right_knee": (24, 26, 28),

            "left_ankle": (25, 27, 31),
            "right_ankle": (26, 28, 32)
        }

        # Connections between landmarks for drawing the skeleton
        self.connections = [
            (11, 12),

            (11, 13), (13, 15),
            (12, 14), (14, 16),

            (11, 23), (12, 24),
            (23, 24),

            (23, 25), (25, 27),
            (24, 26), (26, 28),

            (27, 31), (28, 32),

            (15, 17), (15, 19),
            (16, 18), (16, 20),

            (27, 29), (28, 30)
        ]

    # Store the 33 detected landmarks
    def set_landmarks(self, landmarks):

        self.landmarks = landmarks

        self.coordinates = np.array([
            [lm.x, lm.y, lm.z]
            for lm in landmarks
        ])

        self.visibility = np.array([
            lm.visibility
            for lm in landmarks
        ])

    # Convert normalized coordinates into pixel coordinates
    def get_pixel_coordinates(self, index, frame):

        height, width = frame.shape[:2]

        landmark = self.landmarks[index]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        return (x, y)

    # Check whether a landmark is sufficiently visible
    def is_visible(self, index, threshold=0.5):

        if self.visibility is None:
            return False

        return self.visibility[index] >= threshold

    # Calculate angle between three landmarks
    def calculate_angle(self, index_a, index_b, index_c):

        a = self.coordinates[index_a]
        b = self.coordinates[index_b]
        c = self.coordinates[index_c]

        vector1 = a - b
        vector2 = c - b

        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)

        if magnitude1 == 0 or magnitude2 == 0:
            return None

        cosine = np.dot(vector1, vector2) / (
            magnitude1 * magnitude2
        )

        cosine = np.clip(cosine, -1.0, 1.0)

        angle = np.degrees(np.arccos(cosine))

        return float(angle)

    # Calculate every defined joint angle
    def get_all_angles(self):

        if self.coordinates is None:
            return {}

        angles = {}

        for joint_name, indices in self.joints.items():

            # Ignore angles if any of their landmarks are unclear
            if not all(self.is_visible(i) for i in indices):
                angles[joint_name] = None
                continue

            angles[joint_name] = self.calculate_angle(
                indices[0],
                indices[1],
                indices[2]
            )

        return angles

    # Draw circles and connecting lines
    def draw_skeleton(self, frame):

        if self.landmarks is None:
            return frame

        # Draw connecting lines first
        for start, end in self.connections:

            if self.is_visible(start) and self.is_visible(end):

                point1 = self.get_pixel_coordinates(start, frame)
                point2 = self.get_pixel_coordinates(end, frame)

                cv2.line(
                    frame,
                    point1,
                    point2,
                    (0, 255, 0),
                    2
                )

        # Draw landmark circles
        height, width = frame.shape[:2]

        for index, landmark in enumerate(self.landmarks):

            if self.is_visible(index):

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 0, 255),
                    -1
                )

                # Show landmark index beside each circle
                cv2.putText(
                    frame,
                    str(index),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 0),
                    1
                )

        return frame
