from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import time
from src.wisper import (
    audio_queue,
    transcription_queue,
    start_audio_stream,
    stop_audio_stream,
    process_audio,
    processing_active,
)

app = FastAPI()

# Variable to track the audio stream state
audio_stream_active = False


@app.on_event("startup")
def on_startup():
    """Start the audio stream when FastAPI starts."""
    global audio_stream_active
    start_audio_stream()
    audio_stream_active = True


@app.on_event("shutdown")
def on_shutdown():
    """Stop the audio stream and cleanup when FastAPI shuts down."""
    global audio_stream_active
    stop_audio_stream()
    audio_stream_active = False


@app.get("/transcription")
async def get_transcription():
    """API endpoint to get the latest transcription."""
    global audio_stream_active
    if not audio_stream_active:
        raise HTTPException(status_code=400, detail="Audio stream is not active.")

    def generate_transcription():
        while audio_stream_active:  # Stream only when audio stream is active
            if not transcription_queue.empty():
                transcription = transcription_queue.get()
                yield f"{transcription}\n"
            else:
                time.sleep(0.1)  # Wait briefly to avoid busy-waiting
        yield "Audio stream stopped. No further transcriptions.\n"  # Graceful termination

    return StreamingResponse(generate_transcription(), media_type="text/plain")


@app.get("/start-audio")
def start_audio(background_tasks: BackgroundTasks):
    """Endpoint to start the audio stream explicitly."""
    global audio_stream_active, processing_active
    if audio_stream_active:
        raise HTTPException(status_code=400, detail="Audio stream is already active.")

    start_audio_stream()
    audio_stream_active = True
    # Start audio processing in the background
    background_tasks.add_task(process_audio)
    return {"message": "Audio stream started successfully."}


@app.get("/stop-audio")
def stop_audio():
    """Endpoint to stop the audio stream explicitly."""
    global audio_stream_active, processing_active
    if not audio_stream_active:
        raise HTTPException(status_code=400, detail="Audio stream is not active.")

    # Stop audio streaming and processing
    stop_audio_stream()
    audio_stream_active = False
    processing_active = False

    # Clear queues to prevent leftover data from being processed
    with transcription_queue.mutex:
        transcription_queue.queue.clear()
    with audio_queue.mutex:
        audio_queue.queue.clear()

    return {"message": "Audio stream stopped successfully."}

@app.get("/questions")
async def get_questions():
    """API endpoint to get detected questions."""
    if not audio_stream_active:
        raise HTTPException(status_code=400, detail="Audio stream is not active.")

    def generate_questions():
        while audio_stream_active:
            if not transcription_queue.empty():
                transcription = transcription_queue.get()
                # Check if the transcription contains a detected question
                if "Detected Question:" in transcription:
                    yield f"{transcription}\n"
            else:
                time.sleep(0.1)  # Avoid busy-waiting
        yield "Audio stream stopped. No further questions detected.\n"

    return StreamingResponse(generate_questions(), media_type="text/plain")


@app.get("/")
def main():
    return {"message": "Real-time transcription API is running!"}
