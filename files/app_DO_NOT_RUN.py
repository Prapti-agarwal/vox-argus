# ============================================================================
# ⚠️  DO NOT RUN THIS FILE FOR THE HACKATHON DEMO ⚠️
# ----------------------------------------------------------------------------
# This is an older/alternate backend. It uses a different route (/detect,
# not /predict) and a different response shape than dashboard.html expects,
# AND it doesn't convert webm/ogg mic recordings to WAV before decoding —
# so mic-recorded clips will likely crash it. It also binds the same
# port 5000 as predict_api.py, so running both at once will conflict.
#
# Use predict_api.py instead — it matches dashboard.html and handles
# ffmpeg/pydub conversion for mic recordings.
#
# This file is kept only in case you want its SQLite detection-history
# logging (/history endpoint) ported into predict_api.py later.
# ============================================================================
from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import joblib
import sqlite3
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # allows your HTML/JS frontend to call this server

# ---- Load your trained model (put model.pkl in the same folder as this file) ----
model = joblib.load('model.pkl')

DB_PATH = 'detections.db'


def init_db():
    """Creates the results table if it doesn't already exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            prediction TEXT,
            confidence REAL,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()


init_db()


def extract_mfcc(audio_path):
    """Same 40-dim mean-pooled MFCC fingerprint used during training."""
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean.reshape(1, -1)


@app.route('/detect', methods=['POST'])
def detect():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    temp_path = 'temp_audio.wav'
    audio_file.save(temp_path)

    try:
        features = extract_mfcc(temp_path)
        prediction = model.predict(features)[0]            # 'REAL' or 'FAKE'
        confidence = model.predict_proba(features).max()   # 0.0 - 1.0

        # Save the RESULT only — not the raw audio — to the database
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            'INSERT INTO detections (filename, prediction, confidence, timestamp) '
            'VALUES (?, ?, ?, ?)',
            (audio_file.filename, str(prediction), float(confidence), datetime.now().isoformat())
        )
        conn.commit()
        conn.close()

        return jsonify({
            'prediction': str(prediction),
            'confidence': round(float(confidence) * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)  # cleanup — audio isn't kept after processing


@app.route('/history', methods=['GET'])
def history():
    """Optional: view past detection results — useful for a demo dashboard."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        'SELECT filename, prediction, confidence, timestamp FROM detections '
        'ORDER BY id DESC LIMIT 50'
    ).fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows])


if __name__ == '__main__':
    app.run(port=5000, debug=True)
