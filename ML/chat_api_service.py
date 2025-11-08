import os
import random
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
from ML.API.recommend_api import load_orders, get_popular

load_dotenv()

app = FastAPI(
    title="Canteen Chatbot API",
    description="Dynamic canteen chatbot with menu, stock, specials, sentiment & recommendations."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MENU_PATH = "ML/Data/raw/menu.csv"

def load_menu():
    df = pd.read_csv(MENU_PATH)
    df["available"] = df["available"].astype(str).str.lower().isin(["yes", "true", "1"])
    return df

def get_menu_text():
    df = load_menu()
    return "\n".join([f"- {row['item_name']} — ₹{row['price']}" for _, row in df.iterrows()])

def get_daily_stock():
    df = load_menu()
    return {row["item_name"]: row["available"] for _, row in df.iterrows()}

def get_daily_specials():
    df = load_menu()
    return df.sample(min(2, len(df)))["item_name"].tolist()

def get_popularity_rank():
    df = load_orders()
    pop = get_popular(df, top_n=10)
    return {entry["item_name"]: idx + 1 for idx, entry in enumerate(pop)}

def detect_mood(text):
    t = text.lower()
    if any(k in t for k in ["tired", "sleepy"]): return "tired"
    if any(k in t for k in ["sad", "upset"]): return "sad"
    if any(k in t for k in ["angry", "irritated"]): return "angry"
    if any(k in t for k in ["hungry", "starving"]): return "hungry"
    return None

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
    MODEL = "gemini-2.5-flash"
except Exception:
    gemini_client = None

def build_system_prompt(user_message):
    menu_text = get_menu_text()
    stock = get_daily_stock()
    specials = get_daily_specials()
    pop_rank = get_popularity_rank()
    mood = detect_mood(user_message)
    mood_hint = {
        "tired": "User is tired. Suggest energy boosters like Cold Coffee.",
        "hungry": "User is extremely hungry. Suggest filling meals like Veg Thali or Paneer Thali.",
        "sad": "User is sad. Suggest comfort foods like Maggi or Samosa.",
        "angry": "User is irritated. Suggest quick-served items like Samosa.",
    }.get(mood, "")

    return f"""
You are the official Canteen AI Assistant.

MENU:
{menu_text}

STOCK STATUS:
{stock}

POPULAR ITEMS:
{pop_rank}

TODAY'S SPECIALS:
{specials}

MOOD HINT:
{mood_hint}

RULES:
- For greetings respond friendly.
- For availability check stock.
- For out-of-stock say unavailable.
- For recommendations respond EXACTLY in JSON:
  {{"action": "recommend", "query": "<user message>"}}
- If item not in menu say unavailable.
- Never invent items or prices.
- Keep responses natural and helpful.
"""

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not gemini_client:
        raise HTTPException(503, "AI unavailable")

    system_prompt = build_system_prompt(request.new_message)

    conversation = [
        types.Content(role="user", parts=[types.Part(text=system_prompt)])
    ]

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
        response = gemini_client.models.generate_content(
            model=MODEL,
            contents=conversation
        )

        reply = response.text

        updated_history = request.history + [
            Content(role="user", parts=[Part(text=request.new_message)]),
            Content(role="model", parts=[Part(text=reply)])
        ]

        return ChatResponse(reply=reply, updated_history=updated_history)

    except Exception as e:
        raise HTTPException(500, f"Gemini API Error: {e}")

@app.get("/")
def home():
    return {"msg": "Dynamic Canteen Chatbot ✅ Running!"}
