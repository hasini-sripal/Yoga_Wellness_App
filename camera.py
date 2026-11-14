
import cv2

class camstuff:

    def __init__(self, camera_id=0):
        self.cam = cv2.VideoCapture(camera_id)
        self.frame = None

    def camera(self):

        if not self.cam.isOpened():
            print("Unable to open camera")
            return

        while True:
            ret, self.frame = self.cam.read()

            if not ret:
                print("Failed to get the frames")
                break

            # Flip frame horizontally
            self.frame = cv2.flip(self.frame, 1)

            # Display camera feed
            cv2.imshow("Camera Feed", self.frame)

            # Exit when Q is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.release()

    def release(self):
        self.cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    camera = camstuff()
    camera.camera()

