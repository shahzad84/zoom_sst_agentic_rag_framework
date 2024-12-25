from faster_whisper import WhisperModel
import pyaudio
import numpy as np
import queue
import threading
from transformers import pipeline
import re

# Initialize Whisper model
model = WhisperModel("small.en")

# Audio Stream Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1  # Mono audio
RATE = 16000  # Whisper works best with 16kHz audio
CHUNK = 1024  # Buffer size (streaming in small chunks)
MIN_AUDIO_LENGTH = RATE * 2  # Minimum 2 seconds of audio at 16kHz

# Queues for audio and transcription data
audio_queue = queue.Queue()
transcription_queue = queue.Queue()

# Temporary buffer for audio processing
audio_buffer = []
audio_buffer_lock = threading.Lock()  # Lock for thread-safe buffer access

# Flags to control audio streaming and processing
stream_active = False
processing_active = False

# Load HuggingFace pipeline for question detection
question_pipeline = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")


def extract_questions(text):
    """Detect if a given text contains a question."""
    # Regex to check for explicit question marks (e.g., 'What is...?')
    if "?" in text:
        return text

    # Use HuggingFace model to detect implicit questions
    prediction = question_pipeline(text)
    if prediction[0]["label"] == "QUESTION" and prediction[0]["score"] > 0.8:  # Confidence threshold
        return text
    return None


def audio_callback(in_data, frame_count, time_info, status):
    """Callback function to collect audio chunks."""
    if stream_active:
        audio_queue.put(in_data)
    return (in_data, pyaudio.paContinue)


def start_audio_stream():
    """Start the audio stream."""
    global stream_active, stream
    stream_active = True

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=audio_callback,
    )
    stream.start_stream()


def stop_audio_stream():
    """Stop the audio stream."""
    global stream_active, stream
    stream_active = False
    stream.stop_stream()
    stream.close()


def process_audio():
    """Process audio data for transcription and question detection."""
    global audio_buffer, processing_active

    processing_active = True

    while processing_active:
        try:
            # Get audio data from the queue
            in_data = audio_queue.get(timeout=1)

            # Convert audio data to numpy array
            audio_array = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_buffer.extend(audio_array)

            # Transcribe if buffer size is sufficient
            if len(audio_buffer) >= MIN_AUDIO_LENGTH:
                audio_chunk = np.array(audio_buffer[:MIN_AUDIO_LENGTH])
                audio_buffer = audio_buffer[MIN_AUDIO_LENGTH:]

                segments, _ = model.transcribe(audio_chunk, beam_size=5)
                for segment in segments:
                    transcription = f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}"
                    transcription_queue.put(transcription)

                    # Extract questions and log them
                    question = extract_questions(segment.text)
                    if question:
                        transcription_queue.put(f"Detected Question: {question}")

        except queue.Empty:
            continue