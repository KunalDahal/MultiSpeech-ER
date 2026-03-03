import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class TextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings


class FusionLayer:
    def __init__(self):
        self.text_encoder = TextEncoder()

    def fuse(self, audio_emb, video_emb, text):
        text_emb = self.text_encoder.embed(text)
        fused = torch.cat([audio_emb, video_emb, text_emb], dim=1)
        
        return fused
