from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, Form, APIRouter
from fastapi.responses import StreamingResponse,JSONResponse
from src.decision_logic_rag import decide_and_answer, web_search_duckduckgo
from src.faiss_vectorstorage_chat.load_document import create_faiss_index, load_documents, stored_documents
from src.faiss_vectorstorage_chat.search_faiss_index import search_faiss_index
from src. faiss_vectorstorage_chat.chat_with_document import chat_with_documents
from src.faiss_vectorstorage_chat.data_type import QueryRequest,QueryResponse


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


@app.get("/ask")
async def ask_question(query: str):
    """API endpoint to process a query using RAG, Web Search, or LLM."""
    try:
        response = decide_and_answer(query)
        return JSONResponse({"query": query, "response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/transcription-answer")
async def transcription_to_answer():
    """Stream transcription and process answers using decision logic."""
    global audio_stream_active
    if not audio_stream_active:
        raise HTTPException(status_code=400, detail="Audio stream is not active.")

    def generate_transcription_and_answers():
        while audio_stream_active:
            if not transcription_queue.empty():
                transcription = transcription_queue.get()
                answer = decide_and_answer(transcription)
                yield f"Transcription: {transcription}\nAnswer: {answer}\n"
            else:
                time.sleep(0.1)
        yield "Audio stream stopped. No further processing.\n"

    return StreamingResponse(generate_transcription_and_answers(), media_type="text/plain")

@app.get("/web_search")
def search(query: str):
    """
    FastAPI route to perform a search using the DuckDuckGo API and return the result.
    """
    web_result = web_search_duckduckgo(query)
    return {"web_result": web_result}

@app.post("/upload/")
async def upload_document(file: UploadFile, index_name: str = Form("vector_store")):
    """
    Endpoint to upload a PDF file and create a FAISS index.
    """
    try:
        # Save uploaded file
        file_path = f"./{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Load and split documents
        docs = load_documents(file_path)
        stored_documents.extend(docs)

        # Create FAISS index
        create_faiss_index(docs, index_name=index_name)

        return {"message": f"Index '{index_name}' created successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/chat_with_docs/", response_model=QueryResponse)
async def query_faiss(request: QueryRequest):
    """
    Endpoint to query the FAISS index and generate a response.
    """
    try:
        # Search the index
        indices, _ = search_faiss_index(request.query, index_name=request.index_name, k=request.k)

        # Retrieve matching document chunks
        retrieved_docs = [stored_documents[i] for i in indices[0] if i < len(stored_documents)]

        if not retrieved_docs:
            raise HTTPException(status_code=404, detail="No relevant documents found.")

        # Generate chat response
        response = chat_with_documents(request.query, retrieved_docs)

        return QueryResponse(
            response=response,
            retrieved_docs=[doc.page_content for doc in retrieved_docs]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
    
@app.get("/")
def main():
    return {"message": "Real-time transcription API is running!"}
