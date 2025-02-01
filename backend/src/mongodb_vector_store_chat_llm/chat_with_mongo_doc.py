import requests
import os
from fastapi import HTTPException

# Set the Hugging Face API Key
HF_TOKEN = os.getenv("HF_API_KEY")  # Make sure this is set
if not HF_TOKEN:
    raise HTTPException(status_code=500, detail="Hugging Face API key is missing")

GENERATION_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def clean_context(context: str, max_context_length: int = 1000) -> str:
    """Clean and format the context to remove duplicates and trim length."""
    lines = context.split("\n")
    unique_lines = list(dict.fromkeys(lines))  # Remove duplicates
    cleaned_context = "\n".join(unique_lines)
    return cleaned_context[:max_context_length]

def generate_response(query: str, context: str, max_length: int = 512) -> str:
    """Generate response using Mistral-7B-Instruct via Hugging Face API."""
    context = clean_context(context)
    
    prompt = f"""You are an AI assistant that answers questions based on the provided context. 
    If the context does not contain enough information to answer the question, say "I don't know."

    Context: {context}
    
    Question: {query}
    
    Answer:"""

    try:
        response = requests.post(
            GENERATION_API_URL,
            headers=HEADERS,
            json={
                "inputs": prompt,
                "parameters": {
                    "max_length": max_length,
                    "temperature": 0.3,
                    "repetition_penalty": 1.2,
                    "do_sample": False
                },
                "options": {"wait_for_model": True}
            },
            timeout=30
        )

        print(f"API Response Status Code: {response.status_code}")
        print(f"API Response Content: {response.json()}")  # Debugging

        if response.status_code != 200:
            error_msg = response.json().get("error", "Unknown error from Hugging Face API")
            raise HTTPException(status_code=response.status_code, detail=error_msg)

        # Extract generated response correctly
        response_data = response.json()
        
        # Assuming the response is a list and the first element contains the generated text
        if isinstance(response_data, list) and len(response_data) > 0:
            generated_text = response_data[0].get("generated_text", "").strip()
        else:
            generated_text = response_data.get("generated_text", "").strip()

        if not generated_text:
            raise HTTPException(status_code=500, detail="Empty response from the model")
        
        return generated_text

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# context = """
# The Llama 2 model is a state-of-the-art large language model that can be used for various NLP tasks. 
# It was trained by Meta and has demonstrated impressive capabilities in generating human-like text.
# """
# query = "Who developed Llama 2?"

# response = generate_response(query=query, context=context)

# print("Generated Response:", response)