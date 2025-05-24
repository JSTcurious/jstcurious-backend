from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

class ChatInput(BaseModel):
    message: str

@app.post("/chat")
async def chat(input: ChatInput):
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": input.message}

    response = requests.post(HF_MODEL_URL, headers=headers, json=payload)
    result = response.json()

    if isinstance(result, list) and "generated_text" in result[0]:
        return {"reply": result[0]["generated_text"]}
    return {"reply": "Sorry, something went wrong."}
