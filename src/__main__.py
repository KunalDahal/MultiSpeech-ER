from src.data.loader import extract_audio
from src.speech.whisper import WhisperTranscriber
from src.speech.audio_embed import AudioEmbedder
from src.video.video_embed import VideoEmbedder
from src.emotion.fusion import FusionLayer
from src.emotion.emotion import EmotionModel
import torch

# Define emotion labels (can be customized based on dataset)
EMOTION_LABELS = [
    "happy", "sad", "angry", "fear", "disgust", "surprise", "neutral"
]

def main():
    video_path = "data/harvard.mp4"
    audio_path = "data/harvard.wav"

    print("Step 1: Extracting audio from video...")
    extract_audio(video_path, audio_path)
    print("\nStep 2a: Transcribing audio with Whisper...")
    whisper = WhisperTranscriber()
    text = whisper.transcribe(audio_path)
    print(f"Transcribed text: {text}")

    print("\nStep 2b: Generating audio embeddings...")
    audio_embedder = AudioEmbedder()
    audio_emb = audio_embedder.embed(audio_path)
    print(f"Audio embedding shape: {audio_emb.shape}")

    print("\nStep 3: Generating video embeddings...")
    video_embedder = VideoEmbedder()
    video_emb = video_embedder.embed(video_path)
    print(f"Video embedding shape: {video_emb.shape}")

    print("\nStep 4: Fusing audio, video, and text embeddings...")
    fusion = FusionLayer()
    fused_emb = fusion.fuse(audio_emb, video_emb, text)
    print(f"Fused embedding shape: {fused_emb.shape}")

    print("\nStep 5: Predicting emotion...")
    
    # Calculate input dimension: audio (768) + video (2048) + text (768) = 3584
    input_dim = fused_emb.shape[1]
    num_classes = len(EMOTION_LABELS)
    
    emotion_model = EmotionModel(input_dim=input_dim, num_classes=num_classes)
    emotion_model.eval()
    
    with torch.no_grad():
        emotion_logits = emotion_model(fused_emb)
        predicted_idx = torch.argmax(emotion_logits, dim=1).item()
        predicted_emotion = EMOTION_LABELS[predicted_idx]

    print("\n" + "="*50)
    print("FINAL EMOTION OUTPUT")
    print("="*50)
    print(f"Transcription: {text}")
    print(f"Predicted Emotion: {predicted_emotion}")
    print("="*50)

if __name__ == "__main__":
    main()
