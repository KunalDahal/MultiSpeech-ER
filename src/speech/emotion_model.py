import torch
import torchaudio
import librosa
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

class EmotionRecognizer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "superb/wav2vec2-base-superb-er"
        )
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "superb/wav2vec2-base-superb-er"
        ).to(self.device)
        self.model.eval()

    def predict_emotion(self, audio_path):
        try:
            speech, sr = torchaudio.load(audio_path)
            speech = speech.squeeze().numpy()
        except:
            speech, sr = librosa.load(audio_path, sr=16000)

        if sr != 16000:
            speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
            sr = 16000
        inputs = self.feature_extractor(
            speech,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_id = torch.argmax(logits, dim=-1).item()
        label = self.model.config.id2label[predicted_id]
        
        probs = torch.softmax(logits, dim=-1)
        confidence = probs[0][predicted_id].item()
        
        return label, confidence