# Speech Emotion Recognition

A Python pipeline that transcribes speech and detects emotions from audio segments using Whisper and Wav2Vec2 models.

## Features

- **Speech-to-Text**: Transcribes audio using OpenAI's Whisper model with timestamp segments
- **Emotion Detection**: Classifies speech segments into four emotions (happy, angry, neutral, sad)
- **Confidence Scoring**: Displays prediction confidence percentage for each emotion
- **GPU Acceleration**: Automatically utilizes CUDA if available for faster processing
- **Audio Slicing**: Automatically segments audio based on transcription timestamps
