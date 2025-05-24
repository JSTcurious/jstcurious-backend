from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests

# Load environment variables
load_dotenv()

app = FastAPI()

# Allow requests from all origins (customize in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face API token and model URL
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-alpha"

# Define input schema
class ChatInput(BaseModel):
    message: str

# POST endpoint for chatbot
@app.post("/chat")
async def chat(input: ChatInput):
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {"inputs": input.message}

    # Make request to Hugging Face model
    response = requests.post(HF_MODEL_URL, headers=headers, json=payload)

    try:
        result = response.json()
    except Exception as e:
        return {
            "error": "Failed to parse response",
            "status_code": response.status_code,
            "text": response.text
        }

    # Handle GPT-2 response structure
    if isinstance(result, dict) and "generated_text" in result:
        return {"reply": result["generated_text"]}

    # Handle common list response format (e.g., for other models)
    if isinstance(result, list) and "generated_text" in result[0]:
        return {"reply": result[0]["generated_text"]}

    return {"error": "Unexpected response format", "raw": result}
