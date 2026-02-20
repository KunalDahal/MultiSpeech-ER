import whisper
import torch

def transcribe_with_timestamps(audio_path, model_size="base"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = whisper.load_model(model_size).to(device)

    result = model.transcribe(
        audio_path,
        task="transcribe",
        word_timestamps=False
    )

    return result