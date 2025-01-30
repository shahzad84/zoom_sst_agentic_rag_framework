import os
from dotenv import load_dotenv
import httpx
from fastapi import HTTPException
from httpx import AsyncClient, Timeout

load_dotenv()


async def generate_text(prompt: str) -> str:
    """
    Generate text response using Hugging Face's LLM model.
    """
    # Check if the prompt is an image request
    if "show me" in prompt.lower() or "draw" in prompt.lower():
        return f"Sure! Here's an image of {prompt}."

    # Otherwise, generate a general text response
    headers = {
        "Authorization": f"Bearer {os.getenv('HF_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 50,  # Set the maximum response length
            "temperature": 0.7,  # Optional: Adjust creativity level
        },
        "model": "gpt2",  # Replace with the model you want to use
    }

    async with httpx.AsyncClient(timeout=Timeout(30.0)) as client:
        response = await client.post(
            "https://api-inference.huggingface.co/models/gpt2",
            json=payload,
            headers=headers
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"HTTP Error: {response.status_code} - {response.text}")

        data = response.json()
        return data[0]['generated_text']


