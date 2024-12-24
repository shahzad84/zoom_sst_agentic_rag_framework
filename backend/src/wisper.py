from faster_whisper import WhisperModel
import pyaudio
import numpy as np
import queue
import threading
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
    global audio_buffer, stream_active, audio_queue, transcription_queue, model, audio_buffer_lock

    while stream_active or not audio_queue.empty():
        try:
            in_data = audio_queue.get(timeout=0.1)

            audio_array = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0

            with audio_buffer_lock: # Lock the buffer during modification
                audio_buffer.extend(audio_array)

            # Process the buffer in chunks to avoid memory issues with very long audio.
            chunk_size = RATE * 2 # Process 2-second chunks
            while len(audio_buffer) >= chunk_size:
                with audio_buffer_lock: # Lock the buffer while creating the chunk
                    audio_chunk = np.array(audio_buffer[:chunk_size])
                    audio_buffer = audio_buffer[chunk_size:]
                
                try:
                    segments, info = model.transcribe(audio_chunk, beam_size=5)
                    for segment in segments:
                        transcription_queue.put(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
                        print(f"Transcribed: {segment.text}")

                except Exception as e:
                    print(f"Transcription error: {e}")

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in process_audio: {e}")
            break
