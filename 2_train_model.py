"""
Step 2: Train a classifier on the collected ISL landmark data.

Reads isl_landmark_data.csv (produced by 1_collect_data.py), trains a
Random Forest classifier on the hand-landmark features, evaluates it,
and saves the trained model + label encoder to disk.

Usage:
  python 2_train_model.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATA_CSV = "isl_landmark_data.csv"
MODEL_OUT = "isl_model.joblib"
ENCODER_OUT = "isl_label_encoder.joblib"

def main():
    df = pd.read_csv(DATA_CSV)
    print(f"Loaded {len(df)} samples across {df['label'].nunique()} labels.")
    print(df['label'].value_counts())

    X = df.drop(columns=["label"]).values
    y_raw = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    joblib.dump(clf, MODEL_OUT)
    joblib.dump(encoder, ENCODER_OUT)
    print(f"\nSaved model to {MODEL_OUT}")
    print(f"Saved label encoder to {ENCODER_OUT}")

if __name__ == "__main__":
    main()
