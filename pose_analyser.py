
import numpy as np


class PoseAnalyzer:

    def __init__(self):
        self.landmarks = None

    # Store the landmarks detected by MediaPipe
    def set_landmarks(self, landmarks):
        self.landmarks = landmarks

    # Calculate the angle between three landmarks
    def calculate_angle(self, point_a, point_b, point_c):

        a = np.array(point_a)
        b = np.array(point_b)
        c = np.array(point_c)

        vector1 = a - b
        vector2 = c - b

        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)

        if magnitude1 == 0 or magnitude2 == 0:
            return None

        cosine_angle = np.dot(vector1, vector2) / (
            magnitude1 * magnitude2
        )

        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

        angle = np.degrees(np.arccos(cosine_angle))

        return angle

    # Calculate the distance between two landmarks
    def calculate_distance(self, point_a, point_b):

        a = np.array(point_a)
        b = np.array(point_b)

        distance = np.linalg.norm(a - b)

        return distance

    # Get the coordinates of a particular landmark
    def get_coordinates(self, landmark_index):

        if self.landmarks is None:
            return None

        landmark = self.landmarks[landmark_index]

        return np.array([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    # Calculate a joint angle using landmark indices
    def get_joint_angle(self, index_a, index_b, index_c):

        point_a = self.get_coordinates(index_a)
        point_b = self.get_coordinates(index_b)
        point_c = self.get_coordinates(index_c)

        if point_a is None or point_b is None or point_c is None:
            return None

        return self.calculate_angle(
            point_a,
            point_b,
            point_c
        )
