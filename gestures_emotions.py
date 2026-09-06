"""
combined_input.py
--------------------
Combines gesture_input.py and emotion_input.py into ONE module that
reads the webcam once and detects BOTH the hand gesture AND the facial
emotion at the same time — this matches the original idea of reading
gestures + facial cues together.

Other teammates (like whoever builds app.py) can call:
    get_current_input()
which returns a dictionary like:
    {"gesture": "Yes", "emotion": "Happy"}
so they don't need to touch any of the detection logic themselves.
"""

import os
import time
import urllib.request
from collections import deque, Counter

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------- Model file setup ----------

GESTURE_MODEL_PATH = "gesture_recognizer.task"
GESTURE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)

FACE_MODEL_PATH = "face_landmarker.task"
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

GESTURE_TO_WORD = {
    "Thumb_Up": "Yes",
    "Thumb_Down": "No",
    "Open_Palm": "Stop",
    "Closed_Fist": "Angry",
    "Victory": "Peace",
    "Pointing_Up": "Hello",
    "ILoveYou": "I love you",
}


def ensure_model_downloaded(path, url):
    if not os.path.exists(path):
        print(f"Downloading model to {path} (only happens once)...")
        urllib.request.urlretrieve(url, path)
        print("Model downloaded.")


# ---------- Emotion classification (same logic as emotion_input.py) ----------

def classify_emotion(blendshapes, debug=False):
    scores = {b.category_name: b.score for b in blendshapes}

    smile = max(scores.get("mouthSmileLeft", 0), scores.get("mouthSmileRight", 0))
    frown = max(scores.get("mouthFrownLeft", 0), scores.get("mouthFrownRight", 0))
    brow_up = max(scores.get("browInnerUp", 0), scores.get("browOuterUpLeft", 0))
    jaw_open = scores.get("jawOpen", 0)
    eye_wide = max(scores.get("eyeWideLeft", 0), scores.get("eyeWideRight", 0))
    brow_down = max(scores.get("browDownLeft", 0), scores.get("browDownRight", 0))
    mouth_press = max(scores.get("mouthPressLeft", 0), scores.get("mouthPressRight", 0))
    nose_sneer = max(scores.get("noseSneerLeft", 0), scores.get("noseSneerRight", 0))

    if debug:
        print(
            f"smile={smile:.2f} frown={frown:.2f} brow_up={brow_up:.2f} "
            f"jaw_open={jaw_open:.2f} eye_wide={eye_wide:.2f} "
            f"brow_down={brow_down:.2f} mouth_press={mouth_press:.2f} nose_sneer={nose_sneer:.2f}"
        )
    if smile > 0.4:
        return "Happy"
    elif jaw_open > 0.4 and brow_up > 0.5:
        return "Surprised"
    elif brow_down > 0.4 or nose_sneer > 0.3:
        return "Angry"
    elif frown > 0.06:
        return "Sad"
    elif brow_up > 0.65:
        return "Worried"
    else:
        return "Neutral"


# ---------- Combined detector class ----------

class CombinedInputDetector:
    """
    Wraps both the gesture recognizer and the face landmarker, reading
    from the SAME webcam frame each time, so both run together smoothly.
    """

    def __init__(self):
        ensure_model_downloaded(GESTURE_MODEL_PATH, GESTURE_MODEL_URL)
        ensure_model_downloaded(FACE_MODEL_PATH, FACE_MODEL_URL)

        gesture_options = mp_vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=GESTURE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self.gesture_recognizer = mp_vision.GestureRecognizer.create_from_options(
            gesture_options
        )

        face_options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            num_faces=1,
        )
        self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(
            face_options
        )

        self.cap = cv2.VideoCapture(0)
        self.start_time = time.time()

        # Smoothing buffer, same idea as before — avoids emotion flicker
        self.recent_emotions = deque(maxlen=8)
        self.stable_emotion = "Neutral"

    def read_frame_and_detect(self, debug=False):
        """
        Reads one frame from the webcam and returns:
            frame (for displaying), gesture (str or None), emotion (str)
        """
        success, frame = self.cap.read()
        if not success:
            return None, None, None

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - self.start_time) * 1000)

        # --- Gesture detection ---
        gesture_result = self.gesture_recognizer.recognize_for_video(
            mp_image, timestamp_ms
        )
        gesture_word = None
        if gesture_result.gestures and gesture_result.gestures[0]:
            top_gesture = gesture_result.gestures[0][0]
            if top_gesture.category_name != "None" and top_gesture.score > 0.6:
                gesture_word = GESTURE_TO_WORD.get(
                    top_gesture.category_name, top_gesture.category_name
                )

        # --- Emotion detection ---
        face_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
        raw_emotion = "Neutral"
        if face_result.face_blendshapes:
            raw_emotion = classify_emotion(face_result.face_blendshapes[0], debug=debug)

        self.recent_emotions.append(raw_emotion)
        if len(self.recent_emotions) == self.recent_emotions.maxlen:
            most_common, count = Counter(self.recent_emotions).most_common(1)[0]
            if count >= self.recent_emotions.maxlen // 2:
                self.stable_emotion = most_common

        return frame, gesture_word, self.stable_emotion

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.gesture_recognizer.close()
        self.face_landmarker.close()


# ---------- Simple function other teammates can import and call ----------
# NOTE: this creates a NEW detector + opens the webcam every time it's called,
# which is fine for quick testing but not ideal for a real app loop.
# For app.py, it's better to create ONE CombinedInputDetector and keep
# calling .read_frame_and_detect() on it inside your own loop.

def get_current_input():
    """
    Quick one-shot helper: opens the webcam, reads a few frames to let
    detection stabilize, returns one {"gesture": ..., "emotion": ...} result,
    then closes the webcam again.
    """
    detector = CombinedInputDetector()
    result = {"gesture": None, "emotion": "Neutral"}
    for _ in range(15):  # read a few frames so smoothing has data to work with
        frame, gesture, emotion = detector.read_frame_and_detect()
        if gesture:
            result["gesture"] = gesture
        result["emotion"] = emotion
    detector.close()
    return result


# ---------- Run this file directly to see both detections live ----------

if __name__ == "__main__":
    detector = CombinedInputDetector()
    last_gesture = None
    last_emotion = None

    try:
        while True:
            frame, gesture, emotion = detector.read_frame_and_detect(debug=False)
            if frame is None:
                print("Couldn't access webcam.")
                break

            if gesture != last_gesture and gesture is not None:
                print(f"Gesture: {gesture}")
                last_gesture = gesture
            if emotion != last_emotion:
                print(f"Emotion: {emotion}")
                last_emotion = emotion

            display_text = f"Gesture: {gesture or '...'}  |  Emotion: {emotion}"
            cv2.putText(
                frame, display_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )
            cv2.imshow("Combined Input", frame)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break
    finally:
        detector.close()
