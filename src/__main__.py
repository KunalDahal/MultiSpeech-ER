from src.speech.wishper_transcribe import transcribe_with_timestamps
from src.speech.audio_utils import slice_audio_segments
from src.speech.emotion_model import EmotionRecognizer

AUDIO_FILE = "harvard.wav"

print("Transcribing...")
result = transcribe_with_timestamps(AUDIO_FILE)

print("Full Text:", result["text"])

print("\nSlicing segments...")
segments = slice_audio_segments(
    AUDIO_FILE,
    result["segments"]
)

print("\nDetecting emotions...")
emotion_model = EmotionRecognizer()

for seg in segments:
    emotion = emotion_model.predict_emotion(seg["file"])

    print("Text:", seg["text"])
    print("Emotion:", emotion)
    print("-" * 50)