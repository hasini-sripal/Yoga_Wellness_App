
import cv2
import time


class Camera:

    def __init__(self, camera_id=0):

        self.cap = cv2.VideoCapture(camera_id)

        self.frame = None
        self.fps = 0
        self.previous_time = time.perf_counter()

    def read_frame(self):

        if not self.cap.isOpened():
            print("Unable to open camera.")
            return False, None

        ret, self.frame = self.cap.read()

        if not ret:
            print("Failed to capture frame.")
            return False, None

        # Mirror the camera feed
        self.frame = cv2.flip(self.frame, 1)

        # Calculate FPS
        current_time = time.perf_counter()
        elapsed_time = current_time - self.previous_time

        if elapsed_time > 0:
            self.fps = 1 / elapsed_time

        self.previous_time = current_time

        return True, self.frame

    def release(self):

        self.cap.release()
        cv2.destroyAllWindows()
