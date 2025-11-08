import os
import pandas as pd
import random
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
from ML.API.recommend_api import load_orders, get_popular

load_dotenv()

app = FastAPI(title="Canteen Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MENU_PATH = "ML/Data/raw/menu.csv"

def load_menu():
    df = pd.read_csv(MENU_PATH, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    if "item_name" not in df.columns:
        raise ValueError("menu.csv must contain an 'item_name' column")
    if "price" not in df.columns:
        raise ValueError("menu.csv must contain a 'price' column")
    if "available" not in df.columns:
        df["available"] = True
    df["available"] = df["available"].astype(str).str.lower().isin(["yes", "true", "1"])
    return df

def menu_text():
    df = load_menu()
    return "\n".join([f"- {row['item_name']} — ₹{row['price']}" for _, row in df.iterrows()])

def stock_status():
    df = load_menu()
    return {row["item_name"]: bool(row["available"]) for _, row in df.iterrows()}

def specials():
    df = load_menu()
    if len(df) < 2:
        return df["item_name"].tolist()
    return random.sample(df["item_name"].tolist(), 2)

def popularity_rank():
    orders = load_orders()
    pop = get_popular(orders, top_n=10)
    return {entry["item_name"]: i + 1 for i, entry in enumerate(pop)}

def detect_mood(text):
    t = text.lower()
    if any(x in t for x in ["tired", "sleepy"]): return "tired"
    if any(x in t for x in ["sad", "upset", "down"]): return "sad"
    if any(x in t for x in ["angry", "irritated"]): return "angry"
    if any(x in t for x in ["hungry", "starving"]): return "hungry"
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
except:
    gemini_client = None

def system_prompt(message):
    m = detect_mood(message)
    menu = menu_text()
    stock = stock_status()
    pop = popularity_rank()
    sp = specials()
    mood_hint = {
        "tired": "User is tired. Suggest energy boosters like Cold Coffee.",
        "hungry": "User is hungry. Suggest filling meals like Veg Thali or Paneer Thali.",
        "sad": "User is sad. Suggest comfort foods like Maggi or Samosa.",
        "angry": "User is irritated. Suggest quick items like Samosa."
    }.get(m, "")
    return f"""
You are the intelligent and friendly canteen assistant.

MENU:
{menu}

STOCK:
{stock}

POPULARITY:
{pop}

TODAY_SPECIALS:
{sp}

MOOD_HINT:
{mood_hint}

RULES:
- Respond naturally to greetings.
- Check STOCK strictly before confirming availability.
- If an item is unavailable, say it is not available today.
- If a user asks for recommendations, reply exactly in JSON:
  {{"action":"recommend","query":"<user message>"}}
- Only use menu items and prices from MENU.
- If user asks for a non-menu item, say it is not available.
- Keep responses concise, friendly, and helpful.
"""

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not gemini_client:
        raise HTTPException(503, "AI unavailable")

    prompt = system_prompt(request.new_message)

    convo = [types.Content(role="user", parts=[types.Part(text=prompt)])]

    for msg in request.history:
        convo.append(
            types.Content(
                role=msg.role,
                parts=[types.Part(text=p.text) for p in msg.parts]
            )
        )

    convo.append(types.Content(role="user", parts=[types.Part(text=request.new_message)]))

    try:
        res = gemini_client.models.generate_content(model=MODEL, contents=convo)
        reply = res.text
        updated_history = request.history + [
            Content(role="user", parts=[Part(text=request.new_message)]),
            Content(role="model", parts=[Part(text=reply)])
        ]
        return ChatResponse(reply=reply, updated_history=updated_history)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/")
def root():
    return {"msg": "Chatbot Live ✅"}
