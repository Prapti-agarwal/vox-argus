"""
SIH Project: Voice Cloning Detection API
Wraps the trained model (model.pkl) in a simple web API so a frontend
(website, app, etc.) can send an audio file and get back REAL/FAKE + percentages.

-------------------------------------------------------------------------------
INSTALL
-------------------------------------------------------------------------------
pip install flask flask-cors librosa numpy joblib pydub

You also need ffmpeg installed and on your PATH (pydub shells out to it
to decode webm/ogg/mp3/etc.):
  Windows: download from https://www.gyan.dev/ffmpeg/builds/ (the
           "essentials" build), unzip, and add its bin/ folder to PATH.
           Restart your terminal afterward so PATH changes take effect.
  Mac:     brew install ffmpeg
  Linux:   sudo apt install ffmpeg

-------------------------------------------------------------------------------
RUN
-------------------------------------------------------------------------------
python predict_api.py
    -> starts a local server at http://localhost:5000

-------------------------------------------------------------------------------
HOW YOUR FRIEND'S FRONTEND CALLS IT
-------------------------------------------------------------------------------
POST http://localhost:5000/predict
  - form-data, field name "audio", value = the audio file

Example using JavaScript fetch (what she'd put in the frontend):

    const formData = new FormData();
    formData.append("audio", audioFile);  // audioFile = a File object from an <input type="file">

    fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData,
    })
    .then(res => res.json())
    .then(data => console.log(data));

Response looks like:
    {
      "label": "REAL",
      "real_percent": 94.32,
      "fake_percent": 5.68
    }
"""

import os
import tempfile
import traceback
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# FFmpeg paths
FFMPEG_DIR = r"C:\Users\PRAPTI\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build\bin"

os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

from pydub import AudioSegment

AudioSegment.converter = os.path.join(FFMPEG_DIR, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(FFMPEG_DIR, "ffprobe.exe")
N_MFCC = 40
SAMPLE_RATE = 16000
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")

app = Flask(__name__)
CORS(app)  # allows a frontend running on a different port/domain to call this API

print(f"Loading model from {MODEL_PATH} ...")
bundle = joblib.load(MODEL_PATH)
clf = bundle["model"]
print("Model loaded. API ready.")


def convert_to_wav(src_path):
    """
    Browsers record audio as webm/ogg (Opus), which libsndfile (used by
    librosa) cannot decode directly. Use pydub + ffmpeg to transcode
    whatever format comes in (webm, ogg, mp3, m4a, wav...) into a plain
    WAV file that librosa can always read.
    """
    wav_path = src_path + "_converted.wav"
    audio = AudioSegment.from_file(src_path)  # ffmpeg auto-detects the format
    audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
    audio.export(wav_path, format="wav")
    return wav_path


def extract_features(filepath):
    import librosa
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    if len(y) < sr * 0.1:
        return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfcc, axis=1)


@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No 'audio' file provided in the request"}), 400

    audio_file = request.files["audio"]

    # Save to a temp file so librosa can read it
    suffix = os.path.splitext(audio_file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    wav_path = None
    try:
        # Always transcode first -- handles webm/ogg from the browser
        # recorder as well as wav/mp3 uploads, in one consistent path.
        try:
            wav_path = convert_to_wav(tmp_path)
        except Exception:
            print("ffmpeg/pydub conversion failed:\n" + traceback.format_exc())
            return jsonify({
                "error": "Could not decode that audio file. Make sure ffmpeg "
                         "is installed and on PATH, and try a .wav or .mp3 file."
            }), 400

        feat = extract_features(wav_path)
        if feat is None:
            return jsonify({"error": "Audio file too short or unreadable"}), 400

        feat = feat.reshape(1, -1)
        probs = clf.predict_proba(feat)[0]  # [REAL, FAKE]
        real_pct = round(float(probs[0]) * 100, 2)
        fake_pct = round(float(probs[1]) * 100, 2)
        label = "FAKE" if fake_pct >= real_pct else "REAL"

        return jsonify({
            "label": label,
            "real_percent": real_pct,
            "fake_percent": fake_pct,
        })
    except Exception:
        # Catch-all so the frontend always gets JSON back, never Flask's
        # default HTML 500 page (which breaks res.json() on the frontend
        # and shows up there as a misleading "server unreachable" error).
        print("Unhandled error in /predict:\n" + traceback.format_exc())
        return jsonify({"error": "Something went wrong analyzing that clip."}), 500
    finally:
        os.remove(tmp_path)
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
