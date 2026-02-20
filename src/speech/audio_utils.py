import os
import librosa
import soundfile as sf

def slice_audio_segments(audio_path, segments, output_dir="segments"):
    os.makedirs(output_dir, exist_ok=True)

    y, sr = librosa.load(audio_path, sr=16000)

    sliced_files = []

    for i, seg in enumerate(segments):
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)

        audio_chunk = y[start_sample:end_sample]

        file_path = os.path.join(output_dir, f"segment_{i}.wav")
        sf.write(file_path, audio_chunk, sr)

        sliced_files.append({
            "file": file_path,
            "text": seg["text"]
        })

    return sliced_files