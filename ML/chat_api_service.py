import pandas as pd
from fastapi import APIRouter, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
from ML.API.recommend_api import load_orders, get_popular, get_highest_rated, find_by_category

router = APIRouter()

MENU_PATH = "ML/Data/raw/menu.csv"

def load_menu():
    return pd.read_csv(MENU_PATH)

def menu_text():
    df = load_menu()
    return "\n".join([f"- {row['item_name']} — ₹{row['price']}" for _, row in df.iterrows()])

def stock_status():
    df = load_menu()
    return {row["item_name"]: True for _, row in df.iterrows()}

def specials():
    df = load_menu()
    return df.sample(min(2, len(df)))["item_name"].tolist()

def popularity_rank():
    df = load_orders()
    p = get_popular(df, top_n=10)
    return {v["item_name"]: i + 1 for i, v in enumerate(p)}

def mood_detect(text):
    t = text.lower()
    if "tired" in t or "sleepy" in t: return "tired"
    if "sad" in t or "upset" in t: return "sad"
    if "angry" in t or "irritated" in t: return "angry"
    if "hungry" in t or "starving" in t: return "hungry"
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

def system_prompt(msg):
    m = mood_detect(msg)
    moods = {
        "tired": "User is tired. Suggest Cold Coffee or energizing items.",
        "hungry": "User is hungry. Suggest heavy meals like Thali.",
        "sad": "User is sad. Suggest comfort foods like Maggi.",
        "angry": "User is irritated. Suggest quick food like samosa."
    }
    mood_line = moods.get(m, "")
    return f"""
You are the canteen assistant.

MENU:
{menu_text()}

STOCK:
{stock_status()}

POPULARITY:
{popularity_rank()}

SPECIALS:
{specials()}

MOOD:
{mood_line}

RULES:
- Respond friendly for greetings.
- If user requests recommendation return JSON:
  {{"action":"recommend","query":"{msg}"}}
- If user wants high rating dishes mention items from top-rated list.
- Never invent items.
"""

@router.post("/chat")
async def chat(req: ChatRequest):
    if not gemini_client:
        raise HTTPException(503, "AI offline")

    prompt = system_prompt(req.new_message)
    convo = [types.Content(role="user", parts=[types.Part(text=prompt)])]

    for x in req.history:
        convo.append(types.Content(role=x.role, parts=[types.Part(text=p.text) for p in x.parts]))

    convo.append(types.Content(role="user", parts=[types.Part(text=req.new_message)]))

    try:
        r = gemini_client.models.generate_content(model=MODEL, contents=convo)
        reply = r.text
        new_hist = req.history + [
            Content(role="user", parts=[Part(text=req.new_message)]),
            Content(role="model", parts=[Part(text=reply)])
        ]
        return ChatResponse(reply=reply, updated_history=new_hist)
    except Exception as e:
        raise HTTPException(500, str(e))

@router.get("/")
def home():
    return {"msg": "Chatbot Ready"}
