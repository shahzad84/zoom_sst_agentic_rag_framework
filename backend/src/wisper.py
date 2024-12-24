from faster_whisper import WhisperModel

model=WhisperModel("small.en")


try:
    segments, info = model.transcribe("Recording.mp3")
    for segment in segments:
        print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
except Exception as e:
    print(f"Error during transcription: {e}")