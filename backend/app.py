from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, Form, APIRouter
from fastapi.responses import StreamingResponse,JSONResponse
from src.decision_logic_rag import decide_and_answer, web_search_duckduckgo
from src.faiss_vectorstorage_chat.load_document import create_faiss_index, load_documents, stored_documents
from src.faiss_vectorstorage_chat.search_faiss_index import search_faiss_index
from src. faiss_vectorstorage_chat.chat_with_document import chat_with_documents
from src.faiss_vectorstorage_chat.data_type import QueryRequest,QueryResponse
import time
from loguru import logger
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


# @app.get("/transcription-answer")
# async def transcription_to_answer():
#     """Stream transcription and process answers using decision logic."""
#     global audio_stream_active
#     if not audio_stream_active:
#         raise HTTPException(status_code=400, detail="Audio stream is not active.")

#     def generate_transcription_and_answers():
#         while audio_stream_active:
#             if not transcription_queue.empty():
#                 transcription = transcription_queue.get()
#                 answer = decide_and_answer(transcription)
#                 yield f"Transcription: {transcription}\nAnswer: {answer}\n"
#             else:
#                 time.sleep(0.1)
#         yield "Audio stream stopped. No further processing.\n"

#     return StreamingResponse(generate_transcription_and_answers(), media_type="text/plain")

@app.get("/question-answer")
async def question_answer_stream():
    """Stream detected questions and answers in real-time."""
    if not audio_stream_active:
        raise HTTPException(status_code=400, detail="Audio stream is not active.")

    def generate_answers():
        while audio_stream_active:
            if not transcription_queue.empty():
                transcription = transcription_queue.get()
                if "Detected Question:" in transcription:
                    # Extract the actual question
                    question = transcription.split("Detected Question:")[-1].strip()
                    # Get answer
                    answer = decide_and_answer(question)
                    yield f"Question: {question}\nAnswer: {answer}\n"
            else:
                time.sleep(0.1)
        yield "Audio stream stopped. No further answers.\n"

    return StreamingResponse(generate_answers(), media_type="text/plain")


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
    

from data_type import PromptRequest
from src.react_agent.agent_chat import interpret_prompt, execute_action

@app.post("/react_agent_chat")
def process_prompt(request: PromptRequest):
#     {
#   "prompt": "Send an email to shahzad20022002@gmail.com with subject 'Hello' and body 'How are you?'"
# }

    try:
        # Step 1: Use LLM to interpret the user prompt
        llm_response = interpret_prompt(request.prompt)

        # Step 2: Execute the action based on LLM output
        response = execute_action(llm_response, request.prompt)
        return response
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=400, detail=str(e))



from data_type import ChatRequest
from src.image_chat_llm.chat_with_image import generate_text
from src.image_chat_llm.images import generate_image
from src.image_chat_llm.memory import Memory
import traceback

memory = Memory()
import urllib
@app.post("/chat_with_images")
async def chat_with_images(request: ChatRequest):
    try:
        user_id = request.user_id
        user_message = request.message

        # Generate AI text response
        ai_response = await generate_text(user_message)

        # Check if an image was requested
        if "show me" in user_message.lower() or "draw" in user_message.lower():
            key = f"{user_id}_last_prompt"
            image_prompt = user_message  # Use current message directly

            # Save the prompt for future use
            memory.set(key, image_prompt)

            # Return the AI response and image endpoint URL
            return {
                "response": ai_response,
                "image_url": f"/generate_image?prompt={urllib.parse.quote(image_prompt)}"
            }

        # Return only the AI response
        return {"response": ai_response}

    except Exception as e:
        error_message = f"Error occurred in /chat_with_images endpoint: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)

#using this you can see what llm have generated(use local host and rest of the generated url) :http://localhost:8000/generate_image?prompt=Show%20me%20an%20image%20of%20a%20catfish%20in%20a%20series%20of%20small%20fish%20pots 

# Image generation endpoint
@app.get("/generate_image")
async def generate_image_endpoint(prompt: str):
    return await generate_image(prompt)


# uvicorn app:app --reload

#mongodb_vector_search and chat

from src.mongodb_vector_store_chat_llm.store_chunk_in_mongodb import store_chunks_in_mongodb
import os
from src.mongodb_vector_store_chat_llm.mongodb_vector_search import mongodb_vector_search
from src.mongodb_vector_store_chat_llm.chat_with_mongo_doc import generate_response, clean_context
import uuid
from src.mongodb_vector_store_chat_llm.load_document import summarize_context
from data_type import QueryResponse,QueryRequest

@app.post("/mongodb_upload/")
async def upload_document(file: UploadFile, document_source: str = Form(...)):
    """Endpoint to upload PDF and store in MongoDB"""
    try:
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        docs = load_documents(file_path)
        inserted_ids = store_chunks_in_mongodb(docs, document_source)
        
        os.remove(file_path)
        return {"message": f"Uploaded {len(inserted_ids)} chunks successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query_mongodb/", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Endpoint to query documents and get response"""
    try:
        # Vector search
        results = mongodb_vector_search(request.query, request.k)
        if not results:
            raise HTTPException(status_code=404, detail="No relevant documents found")
        
        # Generate raw context
        raw_context = "\n\n".join([res["text"] for res in results])
        
        # Clean the context (remove duplicates and irrelevant information)
        cleaned_context = clean_context(raw_context, max_context_length=2000)  # Adjust max length as needed
        
        # Summarize the cleaned context (if needed)
        summarized_context = summarize_context(cleaned_context, max_length=1000)
        
        # Generate response
        answer = generate_response(request.query, summarized_context)
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        print(answer)
        return QueryResponse(
            response=answer,
            retrieved_docs=results,  # Return full search results
            session_id=session_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))






