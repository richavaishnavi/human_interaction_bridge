"""
sign_input.py — ISL (Indian Sign Language) recognition module.

This is the "sign" input piece of the Human Interaction Bridge project.
It loads a trained model (see train_sign_model.py) and recognizes ISL
signs from the webcam in real time, emitting the SAME unified signal
format as your other input modules (gesture, emotion, speech):

    {"type": "sign", "value": "<label>", "confidence": 0.87}

Run standalone for testing:
    python sign_input.py

Or import get_sign_signal() / run_sign_recognition() into your main
Streamlit app to plug it in alongside gesture/emotion detection.
"""

import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque, Counter

MODEL_PATH = "isl_model.joblib"
ENCODER_PATH = "isl_label_encoder.joblib"
CONFIDENCE_THRESHOLD = 0.6
SMOOTHING_WINDOW = 8

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def _extract_landmarks(hand_landmarks):
    """Must match normalize_landmarks() in 1_collect_data.py EXACTLY,
    since that's what the model was trained on: wrist-relative
    (translation-invariant) and scaled by hand size (scale-invariant)."""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist = coords[0].copy()
    coords = coords - wrist  # translate so wrist is the origin

    scale = np.linalg.norm(coords[9])  # distance from wrist to middle-finger base
    if scale < 1e-6:
        scale = 1e-6
    coords = coords / scale

    return coords.flatten().reshape(1, -1)


def run_sign_recognition(show_window=True):
    """Runs the webcam loop. Yields a unified signal dict on each
    confidently-recognized, smoothed prediction. Press 'q' to stop
    when show_window=True."""

    clf = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    recent_preds = deque(maxlen=SMOOTHING_WINDOW)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            signal = None

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                if show_window:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                features = _extract_landmarks(hand_landmarks)
                probs = clf.predict_proba(features)[0]
                best_idx = np.argmax(probs)
                confidence = float(probs[best_idx])

                if confidence >= CONFIDENCE_THRESHOLD:
                    predicted_label = encoder.inverse_transform([best_idx])[0]
                    recent_preds.append(predicted_label)
                else:
                    recent_preds.append(None)

                counts = Counter([p for p in recent_preds if p is not None])
                if counts:
                    smoothed_label, freq = counts.most_common(1)[0]
                    signal = {
                        "type": "sign",
                        "value": smoothed_label,
                        "confidence": round(confidence, 2),
                    }

            if show_window:
                text = f"{signal['value']} ({signal['confidence']})" if signal else ""
                cv2.putText(frame, text, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.putText(frame, "q = quit", (10, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.imshow("ISL Sign Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if signal:
                yield signal

    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Starting ISL sign recognition. Press 'q' in the window to quit.\n")
    for sig in run_sign_recognition(show_window=True):
        print(sig)
