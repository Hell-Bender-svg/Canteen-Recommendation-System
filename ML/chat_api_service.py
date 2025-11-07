import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from google import genai
from google.genai import types

load_dotenv()  # Load .env file

# --- Pydantic Models ---
class Part(BaseModel):
    text: str

class Content(BaseModel):
    role: str = Field(..., pattern="^(user|model)$")
    parts: List[Part]

class ChatRequest(BaseModel):
    history: List[Content] = []     # ✅ Allow empty history
    new_message: str

class ChatResponse(BaseModel):
    reply: str
    updated_history: List[Content]

# --- FastAPI App ---
app = FastAPI(
    title="Gemini Conversational API",
    description="Multi-turn chat using Gemini + persistent history."
)

# --- Gemini Client Initialization ---
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing. Create a .env file!")

try:
    gemini_client = genai.Client(api_key=API_KEY)
    GEMINI_MODEL = "gemini-2.5-flash"
except Exception as e:
    gemini_client = None
    print(f"CRITICAL: Failed to initialize Gemini Client. Error: {e}")

# --- Chat Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):

    if not gemini_client:
        raise HTTPException(503, "AI service unavailable")

    # Convert existing history to Gemini types
    contents_for_gemini = [
        types.Content(
            role=msg.role,
            parts=[types.Part(text=p.text) for p in msg.parts]

        )
        for msg in request.history
    ]

    # Add new user message
    contents_for_gemini.append(
        types.Content(
            role="user",
            parts=[types.Part(text=request.new_message)]
        )
    )

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents_for_gemini
        )

        bot_reply = response.text

        # Add the new messages to the returned history
        updated_history = request.history + [
            Content(role="user", parts=[Part(text=request.new_message)]),
            Content(role="model", parts=[Part(text=bot_reply)])
        ]

        return ChatResponse(
            reply=bot_reply,
            updated_history=updated_history
        )

    except Exception as e:
        raise HTTPException(500, f"Gemini API error: {e}")

# Home route
@app.get("/")
def home():
    return {"status": "Chat API running", "use": "/docs to test"}
