import argparse
import cv2
import cv2.aruco as aruco
import numpy as np
import time
import os
from keras.models import load_model

# CONSTANTS
IMG_SIZE = 64  # same as trained model
SIZE = (IMG_SIZE, IMG_SIZE)
# chosen gestures for app
# like - take picture
# stop - portrait mode
# peace - add filter
LABEL_NAMES = ["like", "stop", "peace"]

# COMMAND LINE ARGUMENT PARSER
parser = argparse.ArgumentParser(description="Gesture-controlled camera app")
parser.add_argument(
    "--path", type=str, default=".", help="Path to save captured images"
)
parser.add_argument("--timer", type=int, default=5, help="Countdown timer in seconds")
args = parser.parse_args()


# MAIN CLASS
class CameraApp:

    def __init__(self, model_path, save_path, timer_seconds):

        # user-set params
        self.save_path = save_path
        self.timer_seconds = timer_seconds

        # camera
        self.cap = cv2.VideoCapture(0)  # 0
        self.window_name = "Camera App"
        cv2.namedWindow(self.window_name)

        # model
        self.model = load_model(model_path)

        # aruco detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        # picture mode, countdown
        self.mode = "normal"  # normal, portrait, filter
        self.counting_down = False
        self.countdown_end = None
        self.last_gesture = None

        def preprocess(self, crop):
            # color mode = grayscale (like trained model)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # resize
            resized = cv2.resize(gray, SIZE)
            # turn into numpy array
            arr = np.array(resized).astype("float32") / 255.0

            return arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        def predict(self, crop):
            x = self.preprocess(crop)
            probs = self.model.predict(x, verbose=0)
            idx = np.argmax(probs)

            return LABEL_NAMES[idx], probs[0][idx]

        # aruco board detection
        # code reused from previous assignment
        def detect_board(self, frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)

            # only return source array if 4 markers are visible
            if ids is None or len(ids) < 4:
                return None

            source = [None] * 4
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                if 0 <= marker_id <= 3:
                    source[marker_id] = marker_corners[0].mean(axis=0)

            if any(pt is None for pt in source):
                return None

            # order points by position (TL, TR, BL, BR)
            pts = np.float32(source)
            pts = pts[np.argsort(pts[:, 1])]
            top = pts[:2][np.argsort(pts[:2, 0])]
            bottom = pts[2:][np.argsort(pts[2:, 0])]

            return np.array([top[0], top[1], bottom[0], bottom[1]], dtype=np.float32)

        def apply_mode(self, frame):
            # portrait
            if self.mode == "portrait":
                # blur everything except for the center region (assumes person is centered in the frame)

                # blur the whole frame
                blurred = cv2.GaussianBlur(frame, (51, 51), 0)

                # calculate center region (rectangle)
                h, w = frame.shape[:2]
                cx, cy = w // 2, h // 2
                rw, rh = w // 4, h // 3

                # restore center region
                blurred[cy - rh : cy + rh, cx - rw : cx + rw] = frame[
                    cy - rh : cy + rh, cx - rw : cx + rw
                ]

                return blurred

            # filter - sepia
            elif self.mode == "filter":
                # Sepia filter based on:
                # Filipe Chagas, github.com/FilipeChagasDev
                # https://gist.github.com/FilipeChagasDev/bb63f46278ecb4ffe5429a84926ff812

                # convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # normalize
                normalized_gray = np.array(gray, np.float32) / 255

                # multiply by a solid color
                sepia = np.ones(frame.shape)
                sepia[:, :, 0] *= 153  # B
                sepia[:, :, 1] *= 204  # G
                sepia[:, :, 2] *= 255  # R
                sepia[:, :, 0] *= normalized_gray
                sepia[:, :, 1] *= normalized_gray
                sepia[:, :, 2] *= normalized_gray

                return np.array(sepia, np.uint8)

            return frame  # normal picture mode

        # change modes based on gestures
        # the same gesture can be used to return to normal mode if not already in normal mode
        def handle_gesture(self, gesture):
            if gesture == "like":
                # start countdown only if not already counting down
                if not self.counting_down:
                    self.counting_down = True
                    self.countdown_end = time.time() + self.timer_seconds

            elif gesture == "stop":
                # toggle portrait mode
                if self.mode == "portrait":
                    self.mode = "normal"
                else:
                    self.mode = "portrait"

            elif gesture == "peace":
                # toggle sepia filter
                if self.mode == "filter":
                    self.mode = "normal"
                else:
                    self.mode = "filter"

        def take_photo(self, frame):
            # apply current mode to the saved photo 
            photo = self.apply_mode(frame)

            # build filename with timestamp to avoid overwriting
            filename  = f'photo_{int(time.time())}.jpg'
            full_path = os.path.join(self.save_path, filename)

            # make sure the directory exists
            os.makedirs(self.save_path, exist_ok=True)

            cv2.imwrite(full_path, photo)
            print(f'Photo saved to {full_path}')

        def run(self):
            pass  # to be implemented

        def stop(self):
            self.cap.release()
            cv2.destroyAllWindows()


# MAIN
if __name__ == "__main__":
    app = CameraApp(
        model_path="03-camera-app/model_grayscale.keras",
        save_path=args.path,
        timer_seconds=args.timer,
    )
    app.run()
