# Vox Argus

**AI-powered real-time detection of cloned and AI-generated voices across Indian languages.**

Vox Argus lets you upload or record a voice clip and tells you whether it is a **real human voice** or **AI-generated (fake)**, with a confidence percentage for each. It was built for the Smart India Hackathon problem statement *"AI-Powered Real-Time Detection & Prevention of Voice Cloning Impersonation Attacks"*.

## Demo

**Demo video:** 

https://github.com/user-attachments/assets/45cae4c4-23da-4faf-a397-d032a33dae64



| Setup | Permissions |

| ![Setup](screenshots/01-setup.png) | ![Permissions](screenshots/02-permissions.png) |

| Check a call | Result |

| ![Check a call](screenshots/03-check-a-call.png) | ![Result](screenshots/04-result.png) |

## Features

- Classifies audio as **REAL** or **FAKE** with confidence percentages
- Works on **multiple Indian languages** (Hindi, Bengali, Tamil, Marathi, Punjabi) plus English
- Accepts uploaded files (wav, mp3, etc.) and browser recordings (webm/ogg)
- Simple web interface: onboarding flow (name, phone, OTP, permissions) and a detection dashboard
- Record from the microphone or upload a clip, with a live server-status indicator
- Small Flask API that any frontend can call

## How it works

1. Audio is converted to 16 kHz mono WAV with ffmpeg.
2. 40 **MFCC** features are extracted with librosa and averaged over time.
3. A **Random Forest** classifier (200 trees, balanced class weights) predicts the probability of real vs. fake.
4. The Flask API returns the result as JSON.

```json
{
  "label": "REAL",
  "real_percent": 94.32,
  "fake_percent": 5.68
}
```

## Results

Trained on 67,355 audio samples (11,079 real, 56,276 fake):

| Metric | Score |
|---|---|
| Test accuracy | 97.34% |
| Balanced accuracy | 95.35% |

These numbers come from a held-out 20% split of my own dataset, not from an external benchmark such as ASVspoof, so treat them as a prototype result.

## Tech stack

Python, Flask, librosa, scikit-learn, pydub + ffmpeg, HTML / CSS / JavaScript

## Data

The model was trained on a mix of real and synthetic speech:

- Real speech: Google [FLEURS](https://huggingface.co/datasets/google/fleurs) and other recordings
- Synthetic speech: [IndicSynth](https://huggingface.co/datasets/vdivyasharma/IndicSynth),[MLAAD](https://huggingface.co/datasets/mueller91/MLAAD) (Multi-Language Audio Anti-Spoofing Dataset) and TTS-generated clips
- Additional datasets from [Hugging Face](https://huggingface.co/datasets)

The audio data is not included in this repository because of its size and dataset licenses.

## Run it locally

**Requirements:** Python 3.9+ and [ffmpeg](https://ffmpeg.org/download.html) installed and on your PATH.

```bash
git clone https://github.com/Prapti-agarwal/vox-argus.git
cd vox-argus
pip install -r requirements.txt
python predict_api.py
```

Then open **http://localhost:5000** in your browser.

If ffmpeg is installed but not on your PATH, set `FFMPEG_DIR` to its `bin` folder before running.

## Project structure

```
predict_api.py        Flask API + serves the web pages
model.pkl             Trained Random Forest model
train_final_model.py  Feature extraction and training script
model_metrics.json    Saved training metrics
setup.html            Onboarding page
dashboard.html        Detection dashboard
requirements.txt      Python dependencies
```

## Retraining

Put your audio in folders named like `hi real`, `hi fake`, `bn real`, `bn fake` next to `train_final_model.py`, then run:

```bash
pip install tqdm
python train_final_model.py --retrain
```

## My role

I worked on the AI side of the project: dataset preparation, MFCC feature extraction, and training and evaluating the detection model.

## Limitations

- Uses simple averaged MFCC features, so it can miss high-quality clones from newer voice models.
- Real-speech recall is lower than overall accuracy because the dataset has far more fake samples than real ones.
- Not validated on external benchmarks yet.

## Author

[Prapti Agarwal](https://github.com/Prapti-agarwal)
