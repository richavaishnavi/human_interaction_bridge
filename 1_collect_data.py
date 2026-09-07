"""
Step 1 (FIXED): Collect training data for ISL/ASL sign recognition.

FIX: landmarks are now normalized relative to the wrist (translation-
invariant) and scaled by hand size (scale-invariant). This means the
model learns the actual SHAPE of your hand, not where your hand
happened to be positioned on screen when you recorded it. This was
the root cause of signs being confused with each other.

Controls:
  s = save current frame's landmarks as a sample for the current label
  n = move to next label
  q = quit and save CSV

Usage:
  python 1_collect_data.py
"""

import cv2
import mediapipe as mp
import csv
import os
import numpy as np

# ----- CONFIG: edit this list to the signs you want to train -----
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["Yes","No","Eat","love","I Love You","Drink","hello","Code2Create"]
OUTPUT_CSV = "isl_landmark_data.csv"
# -------------------------------------------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def normalize_landmarks(hand_landmarks):
    """Makes landmarks relative to the wrist (position-independent) and
    scaled by hand size (distance-from-camera-independent)."""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist = coords[0].copy()
    coords = coords - wrist  # translate so wrist is the origin

    scale = np.linalg.norm(coords[9])  # distance from wrist to middle-finger base
    if scale < 1e-6:
        scale = 1e-6
    coords = coords / scale

    return coords.flatten().tolist()


def main():
    file_exists = os.path.isfile(OUTPUT_CSV)
    csv_file = open(OUTPUT_CSV, mode="a", newline="")
    csv_writer = csv.writer(csv_file)

    if not file_exists:
        header = ["label"]
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        csv_writer.writerow(header)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    label_idx = 0
    sample_count = 0

    print("Controls: 's' = save sample | 'n' = next label | 'q' = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        features = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            features = normalize_landmarks(hand_landmarks)

        current_label = LABELS[label_idx]
        cv2.putText(frame, f"Label: {current_label}  Samples: {sample_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "s=save  n=next label  q=quit",
                    (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "Move hand around the frame between saves!",
                    (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        cv2.imshow("ISL Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and features is not None:
            csv_writer.writerow([current_label] + features)
            sample_count += 1
        elif key == ord('n'):
            label_idx = (label_idx + 1) % len(LABELS)
            sample_count = 0
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print(f"Data saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
