import torch
import torchaudio
import librosa
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

class EmotionRecognizer:
    def __init__(self):
        print("[DEBUG] Initializing EmotionRecognizer...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[DEBUG] Using device: {self.device}")

        print("[DEBUG] Loading Wav2Vec2 feature extractor...")
        # Use FeatureExtractor instead of Processor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "superb/wav2vec2-base-superb-er"
        )
        print("[DEBUG] Feature extractor loaded successfully")

        print("[DEBUG] Loading Wav2Vec2 model...")
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "superb/wav2vec2-base-superb-er"
        ).to(self.device)
        self.model.eval()
        print(f"[DEBUG] Model loaded successfully with {sum(p.numel() for p in self.model.parameters()):,} parameters")

    def predict_emotion(self, audio_path):
        print(f"[DEBUG] Predicting emotion for: {audio_path}")
        
        print(f"[DEBUG] Loading audio file with torchaudio...")
        try:
            # Try loading with torchaudio first
            speech, sr = torchaudio.load(audio_path)
            speech = speech.squeeze().numpy()
        except:
            # Fall back to librosa if torchaudio fails
            print("[DEBUG] Torchaudio failed, trying librosa...")
            speech, sr = librosa.load(audio_path, sr=16000)
        
        print(f"[DEBUG] Audio loaded: duration={len(speech)/sr:.2f}s, sample_rate={sr}Hz")

        # Ensure correct sample rate
        if sr != 16000:
            print(f"[DEBUG] Resampling from {sr}Hz to 16000Hz")
            speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
            sr = 16000

        print("[DEBUG] Processing audio for model input...")
        inputs = self.feature_extractor(
            speech,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        print("[DEBUG] Moving inputs to device")

        print("[DEBUG] Running inference...")
        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_id = torch.argmax(logits, dim=-1).item()
        label = self.model.config.id2label[predicted_id]
        
        # Get confidence scores
        probs = torch.softmax(logits, dim=-1)
        confidence = probs[0][predicted_id].item()
        
        print(f"[DEBUG] Prediction complete - Emotion: {label}, Confidence: {confidence:.2%}")
        
        return label