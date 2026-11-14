
import cv2

from features import JOINTS, landmarks_to_features


class PoseAnalyzer:

    def __init__(self):

        self.landmarks = None
        self.coordinates = None
        self.visibility = None

        self.joints = JOINTS

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

    def set_landmarks(self, landmarks):

        self.landmarks = landmarks

        self.visibility = [
            lm.visibility for lm in landmarks
        ]

    def is_visible(self, index, threshold=0.5):

        if self.visibility is None:
            return False

        return self.visibility[index] >= threshold

    # Get pixel location of a landmark
    def get_pixel_coordinates(self, index, frame):

        height, width = frame.shape[:2]

        landmark = self.landmarks[index]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        return x, y

    # Get locations of all 33 landmarks
    def get_all_locations(self, frame):

        if self.landmarks is None:
            return {}

        locations = {}

        for index in range(len(self.landmarks)):

            if self.is_visible(index):

                locations[index] = self.get_pixel_coordinates(
                    index, frame
                )

        return locations

    # Calculate all 10 joint angles
    # Uses the same feature calculations as model training
    def get_all_angles(self, width, height):

        if self.landmarks is None:
            return {}

        features = landmarks_to_features(
            self.landmarks,
            width,
            height
        )

        if features is None:
            return {}

        angles = {}

        # First 99 features are coordinates.
        # The remaining 10 features are angles.
        angle_values = features[99:]

        for index, joint_name in enumerate(self.joints):

            landmark_indices = self.joints[joint_name]

            if not all(
                self.is_visible(i)
                for i in landmark_indices
            ):
                angles[joint_name] = None
            else:
                angles[joint_name] = float(angle_values[index])

        return angles

    # Draw body skeleton
    def draw_skeleton(self, frame):

        if self.landmarks is None:
            return frame

        # Draw connecting lines
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

        # Draw all visible landmark circles
        locations = self.get_all_locations(frame)

        for index, (x, y) in locations.items():

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 0, 255),
                -1
            )

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
