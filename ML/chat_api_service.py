import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from google import genai
from google.genai import types
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Gemini Conversational API",
    description="Canteen-aware chatbot with menu."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

CANTEEN_MENU = """
Available items in our University Canteen:

- Veg Sandwich — ₹40  
- Cheese Maggi — ₹60  
- Veg Thali — ₹80  
- Paneer Thali — ₹110  
- Cold Coffee — ₹50  
- Masala Dosa — ₹70  
- Samosa — ₹15  
- Chole Bhature — ₹65  
- Idli Sambhar — ₹40  
- Fried Rice — ₹70  
"""

SYSTEM_PROMPT = f"""
You are the official AI assistant for the College Canteen.

RULES:
1. You ONLY use the following menu:
{CANTEEN_MENU}

2. If a user asks for recommendations:
   Respond EXACTLY in JSON:
   {{"action": "recommend", "query": "<user message>"}}

3. If the user asks about a food item NOT in the menu:
   Respond: "Sorry, that item is not available in our canteen."

4. For price, ingredients, availability: answer using ONLY the menu.

5. Never invent dishes or extra details.
"""

class Part(BaseModel):
    text: str

class Content(BaseModel):
    role: str = Field(..., pattern="^(user|model)$")
    parts: List[Part]

class ChatRequest(BaseModel):
    history: List[Content]
    new_message: str

class ChatResponse(BaseModel):
    reply: str
    updated_history: List[Content]

try:
    gemini_client = genai.Client()
    GEMINI_MODEL = "gemini-2.5-flash"
except Exception:
    gemini_client = None

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):

    if not gemini_client:
        raise HTTPException(503, "AI service unavailable")

    conversation = []

    conversation.append(
        types.Content(
            role="user",
            parts=[types.Part(text=SYSTEM_PROMPT)]
        )
    )

    for msg in request.history:
        conversation.append(
            types.Content(
                role=msg.role,
                parts=[types.Part(text=p.text) for p in msg.parts]
            )
        )

    conversation.append(
        types.Content(
            role="user",
            parts=[types.Part(text=request.new_message)]
        )
    )

    try:
        ai_response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=conversation
        )

        reply_text = ai_response.text

        updated_history = request.history + [
            Content(role="user", parts=[Part(text=request.new_message)]),
            Content(role="model", parts=[Part(text=reply_text)])
        ]

        return ChatResponse(reply=reply_text, updated_history=updated_history)

    except Exception as e:
        raise HTTPException(500, f"Gemini API error: {e}")

@app.get("/")
def home():
    return {"message": "Canteen Chatbot API running ✅", "docs": "/chat/docs"}
