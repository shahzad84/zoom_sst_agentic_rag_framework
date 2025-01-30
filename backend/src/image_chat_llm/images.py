import asyncio
from huggingface_hub import InferenceClient
from PIL import Image
from fastapi.responses import StreamingResponse
import io
import base64
from fastapi import HTTPException

import os
# Initialize the Inference Client with your token
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise ValueError("Hugging Face API key not found in environment variables.")
client = InferenceClient(token=HF_API_KEY)

# Function to generate image
async def generate_image(image_prompt: str):
    try:
        # Run blocking call in thread pool
        image = await asyncio.to_thread(
            client.text_to_image, 
            image_prompt, 
            model="stabilityai/stable-diffusion-3.5-large"
        )
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        
        return StreamingResponse(buffered, media_type="image/png")

    except Exception as e:
        print(f"Error generating image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")


