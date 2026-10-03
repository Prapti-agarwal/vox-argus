import os
import numpy as np
import soundfile as sf
import torch
from transformers import VitsModel, AutoTokenizer

output_root = "dataset"  # or "." if running from inside VoxArgus AI

languages = {
    "te": "facebook/mms-tts-tel",
    "pn": "facebook/mms-tts-pan",
}

# A handful of varied sentences per language — more sentences = more variety in the fake set
sentences = {
    "te": [
        "నమస్కారం, మీరు ఎలా ఉన్నారు?",
        "ఈ రోజు వాతావరణం చాలా బాగుంది.",
        "దయచేసి మీ బ్యాంకు వివరాలు తెలియజేయండి.",
        "మేము మీ ఖాతాను ధృవీకరించాలి.",
        "వెంటనే డబ్బు పంపండి లేకపోతే మీ ఖాతా బ్లాక్ అవుతుంది.",
    ],
    "pn": [
        "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?",
        "ਅੱਜ ਮੌਸਮ ਬਹੁਤ ਵਧੀਆ ਹੈ।",
        "ਕਿਰਪਾ ਕਰਕੇ ਆਪਣੇ ਬੈਂਕ ਵੇਰਵੇ ਦੱਸੋ।",
        "ਸਾਨੂੰ ਤੁਹਾਡਾ ਖਾਤਾ ਪ੍ਰਮਾਣਿਤ ਕਰਨਾ ਪਵੇਗਾ।",
        "ਤੁਰੰਤ ਪੈਸੇ ਭੇਜੋ ਨਹੀਂ ਤਾਂ ਖਾਤਾ ਬਲੌਕ ਹੋ ਜਾਵੇਗਾ।",
    ],
}

for lang_code, model_name in languages.items():
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = VitsModel.from_pretrained(model_name)
    model.eval()

    fake_dir = os.path.join(output_root, f"{lang_code} fake")
    os.makedirs(fake_dir, exist_ok=True)

    count = 0
    # Repeat + slightly vary sentence order to generate ~200 clips per language
    for rep in range(40):
        for i, text in enumerate(sentences[lang_code]):
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                output = model(**inputs).waveform
            audio = output.squeeze().numpy()
            sf.write(os.path.join(fake_dir, f"{lang_code}_fake_{count}.wav"), audio, model.config.sampling_rate)
            count += 1

    print(f"  saved {count} fake clips -> {fake_dir}")

print("Done — Telugu and Punjabi fake audio ready.")