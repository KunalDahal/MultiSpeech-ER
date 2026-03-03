import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio

class AudioEmbedder:
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        self.model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )

    def embed(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)

        inputs = self.processor(
            waveform.squeeze().numpy(),
            sampling_rate=sr,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings