"""
SIH Project: AI-Powered Real-Time Detection & Prevention of Voice Cloning
Impersonation Attacks — Final model (MFCC + Random Forest).

WORKS DIRECTLY WITH THE "RUN" BUTTON — no command-line arguments required.
By default it uses the SAME FOLDER THIS SCRIPT IS SAVED IN as the dataset
root, so you never need to move your audio data out of VoxArgus AI (or
wherever this file lives).

-------------------------------------------------------------------------------
DATASET LAYOUT EXPECTED (auto-detected, no setup needed)
-------------------------------------------------------------------------------
Put this script directly inside the folder that contains your language
folders, e.g.:

    VoxArgus AI/
      train_final_model.py   <- this file
      hi real/   *.mp3
      hi fake/   *.mp3
      bn real/   *.mp3
      bn fake/   *.mp3
      mr real/   *.mp3
      mr fake/   *.mp3
      ta real/   *.mp3
      ta fake/   *.mp3
      pn real/   *.mp3   (no fake yet -> auto-excluded from training)
      te real/   *.mp3   (no fake yet -> auto-excluded from training)

Folder names just need a language code followed by "real" or "fake"
(case-insensitive) as their own word — e.g. "hi real", "HI_FAKE" both work.
Nested layout (dataset/<language>/real/, dataset/<language>/fake/) is also
still supported.

-------------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------------
Just click Run (or: python train_final_model.py)
    -> Trains automatically using audio found next to this script.
       Skips training if a model already exists (model.pkl).

python train_final_model.py "path\to\audio.wav"
    -> Ensures the model exists (trains if needed), then classifies that
       one file and prints REAL / FAKE as percentages.

python train_final_model.py "path\to\a\folder"
    -> Classifies every audio file inside that folder.

Optional flags for advanced use:
    python train_final_model.py --data_dirs "C:\other\dataset" --out model.pkl
    python train_final_model.py --retrain          (force retraining)
"""

import argparse
import os
import sys
import glob
import json
import numpy as np

N_MFCC = 40
SAMPLE_RATE = 16000
RANDOM_STATE = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "model.pkl")
DEFAULT_METRICS_PATH = os.path.join(SCRIPT_DIR, "model_metrics.json")


# -------------------------------------------------------------------------
# Data discovery — supports flat "<code> real"/"<code> fake" AND nested
# <language>/real, <language>/fake layouts.
# -------------------------------------------------------------------------
def find_audio_files(roots):
    exts = ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a")
    pairs = []
    lang_counts = {}
    skipped = 0

    for root in roots:
        for ext in exts:
            for path in glob.glob(os.path.join(root, "**", ext), recursive=True):
                norm = path.replace("\\", "/")
                parts = norm.split("/")
                label = None
                lang_code = None

                for part in parts[:-1]:
                    p_lower = part.lower().strip()
                    tokens = p_lower.split()
                    if "fake" in tokens or "spoof" in tokens:
                        label = 1
                        lang_code = tokens[0] if len(tokens) > 1 else None
                        break
                    elif "real" in tokens or "bonafide" in tokens or "genuine" in tokens:
                        label = 0
                        lang_code = tokens[0] if len(tokens) > 1 else None
                        break

                if label is None:
                    skipped += 1
                    continue

                pairs.append((path, label))
                lang_code = lang_code or "unknown"
                lang_counts.setdefault(lang_code, {"real": 0, "fake": 0})
                lang_counts[lang_code]["real" if label == 0 else "fake"] += 1

    if skipped:
        print(f"[warn] skipped {skipped} files — could not infer real/fake from folder name")

    if lang_counts:
        print("\nPer-language file counts:")
        for lang, counts in sorted(lang_counts.items()):
            flag = "  <- no fake data yet, excluded from training" if counts["fake"] == 0 else ""
            print(f"  {lang:6s} real={counts['real']:5d}  fake={counts['fake']:5d}{flag}")
        print()

    return pairs


def extract_features(filepath):
    import librosa
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    if len(y) < sr * 0.1:
        return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfcc, axis=1)


def build_dataset(pairs):
    from tqdm import tqdm
    X, y, failed = [], [], []
    for path, label in tqdm(pairs, desc="Extracting MFCC features"):
        try:
            feat = extract_features(path)
            if feat is None:
                failed.append(path)
                continue
            X.append(feat)
            y.append(label)
        except Exception:
            failed.append(path)
    if failed:
        print(f"[warn] failed to process {len(failed)} files (corrupt/unreadable) — excluded")
    return np.array(X), np.array(y)


# -------------------------------------------------------------------------
# Train
# -------------------------------------------------------------------------
def train_model(data_dirs, out_path):
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score,
            classification_report, confusion_matrix
        )
        import joblib
    except ImportError as e:
        print("\n[error] Could not import a required package:", e)
        print("If you see a 'DLL load failed' error and this project folder is")
        print("inside OneDrive, that is usually a Windows security policy blocking")
        print("compiled files in cloud-synced folders — not a bug in this script.")
        print("Fix: move this whole folder to a plain local path (e.g. C:\\SIH-Project)")
        print("and run it from there instead.")
        sys.exit(1)

    pairs = find_audio_files(data_dirs)
    if not pairs:
        print("[error] no labeled audio files found next to this script.")
        print("Make sure folders like 'hi real', 'hi fake', etc. are in the same")
        print("directory as train_final_model.py, or pass --data_dirs explicitly.")
        sys.exit(1)

    n_real = sum(1 for _, l in pairs if l == 0)
    n_fake = sum(1 for _, l in pairs if l == 1)
    print(f"Found {len(pairs)} files total -> {n_real} real, {n_fake} fake")

    X, y = build_dataset(pairs)
    print(f"Feature matrix: {X.shape}, labels: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    print("\n===== RESULTS =====")
    print(f"Test Accuracy:          {acc:.4f}")
    print(f"Balanced Accuracy:      {bal_acc:.4f}")
    print("\nClassification report (0=REAL, 1=FAKE):")
    print(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    bundle = {
        "model": clf,
        "n_mfcc": N_MFCC,
        "sample_rate": SAMPLE_RATE,
        "label_map": {0: "REAL", 1: "FAKE"},
    }
    joblib.dump(bundle, out_path)
    print(f"\nSaved model -> {out_path}")

    metrics = {
        "n_samples": len(pairs), "n_real": n_real, "n_fake": n_fake,
        "test_accuracy": acc, "balanced_accuracy": bal_acc,
    }
    metrics_path = os.path.splitext(out_path)[0] + "_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics -> {metrics_path}")

    return bundle


def ensure_model(out_path, data_dirs, retrain=False):
    import joblib
    if not retrain and os.path.exists(out_path):
        print(f"Using existing model -> {out_path}")
        print("(pass --retrain to force retraining)\n")
        return joblib.load(out_path)
    return train_model(data_dirs, out_path)


# -------------------------------------------------------------------------
# Predict — prints REAL/FAKE as percentages, not just the winning label
# -------------------------------------------------------------------------
def predict_one(bundle, filepath):
    clf = bundle["model"]
    feat = extract_features(filepath)
    if feat is None:
        return None
    feat = feat.reshape(1, -1)
    probs = clf.predict_proba(feat)[0]  # class order matches clf.classes_ -> [0, 1] = [REAL, FAKE]
    real_pct = probs[0] * 100
    fake_pct = probs[1] * 100
    label = "FAKE" if fake_pct >= real_pct else "REAL"
    return label, real_pct, fake_pct


def predict_path(bundle, target):
    targets = []
    if os.path.isdir(target):
        for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"):
            targets.extend(glob.glob(os.path.join(target, "**", ext), recursive=True))
    elif os.path.isfile(target):
        targets = [target]
    else:
        print(f"[error] '{target}' is not a valid file or folder.")
        sys.exit(1)

    if not targets:
        print(f"[warn] no audio files found in '{target}'")
        return

    print("\n===== PREDICTIONS =====")
    for t in targets:
        result = predict_one(bundle, t)
        if result is None:
            print(f"{t} -> [could not process]")
            continue
        label, real_pct, fake_pct = result
        print(f"{os.path.basename(t)} -> {label}   |   Real: {real_pct:5.2f}%   Fake: {fake_pct:5.2f}%")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="SIH voice cloning detection — trains automatically, "
                     "predicts REAL/FAKE percentages for any audio file."
    )
    parser.add_argument(
        "target", nargs="?", default=None,
        help="Optional: path to an audio file or a folder of audio files to classify. "
             "If omitted, the script just makes sure a trained model exists."
    )
    parser.add_argument(
        "--data_dirs", nargs="+", default=None,
        help="Folder(s) to train from. Defaults to the folder this script is saved in."
    )
    parser.add_argument("--out", default=DEFAULT_MODEL_PATH, help="Where to save/load the model.")
    parser.add_argument("--retrain", action="store_true", help="Force retraining even if a model already exists.")
    args = parser.parse_args()

    data_dirs = args.data_dirs or [SCRIPT_DIR]

    bundle = ensure_model(args.out, data_dirs, args.retrain)

    if args.target:
        predict_path(bundle, args.target)
    else:
        print("Model is ready.")
        print(f"  -> {args.out}")
        print("\nTo classify a file, run:")
        print(f'  python "{os.path.basename(__file__)}" "path\\to\\audio.wav"')
        print("To classify every file in a folder, run:")
        print(f'  python "{os.path.basename(__file__)}" "path\\to\\folder"')


if __name__ == "__main__":
    main()
