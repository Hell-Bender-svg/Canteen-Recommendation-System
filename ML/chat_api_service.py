import os
import random
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ML.API.recommend_api import load_orders, get_menu, get_popular, get_highest_rated, find_by_category

load_dotenv()

app = FastAPI(title="Canteen Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _lazy_df() -> pd.DataFrame:
    return load_orders()

def _menu_text() -> str:
    df = get_menu(_lazy_df())
    lines = [f"- {r.item_name} — ₹{r.price:.2f} ({r.category})" for r in df.itertuples(index=False)]
    return "\n".join(lines)

def _specials(k: int = 2) -> List[str]:
    df = get_menu(_lazy_df())
    items = df["item_name"].tolist()
    if not items:
        return []
    k = min(k, len(items))
    return random.sample(items, k=k)

def _pop_rank() -> Dict[str, int]:
    pop = get_popular(_lazy_df(), top_n=10)
    return {r["item_name"]: i + 1 for i, r in enumerate(pop)}

def _top_rated(n: int = 10) -> List[Dict]:
    return get_highest_rated(_lazy_df(), top_n=n)

def _detect_intent(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["highest rated", "highly rated", "top rated", "best rated", "high rating"]):
        return "highest_rated"
    if any(k in t for k in ["popular", "trending", "most ordered", "bestseller"]):
        return "popular"
    if any(k in t for k in ["recommend", "suggest", "what should i eat", "what to eat"]):
        return "recommend"
    if any(k in t for k in ["category:", "show category", "show me", "want", "craving"]):
        return "category_maybe"
    return "chat"

def _extract_category(text: str) -> Optional[str]:
    t = text.lower()
    hints = ["snacks", "beverage", "beverages", "lunch", "breakfast", "noodles", "pizza", "thali", "roll", "sandwich", "tea", "coffee", "special"]
    for h in hints:
        if h in t:
            if h == "beverages":
                return "Beverage"
            return h.capitalize()
    return None

class Part(BaseModel):
    text: str

class Content(BaseModel):
    role: str = Field(..., pattern="^(user|model)$")
    parts: List[Part]

class ChatRequest(BaseModel):
    history: List[Content] = []
    new_message: str

class ChatResponse(BaseModel):
    reply: str
    updated_history: List[Content]

def _format_list(items: List[Dict], key: str, second: Optional[str] = None, label: Optional[str] = None) -> str:
    lines = []
    for i, r in enumerate(items, 1):
        if second and second in r:
            lines.append(f"{i}. {r[key]} ({label or second}: {r[second]})")
        else:
            lines.append(f"{i}. {r[key]}")
    return "\n".join(lines) if lines else "No items found."

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        intent = _detect_intent(request.new_message)
        if intent == "highest_rated":
            items = _top_rated(8)
            reply = "Top rated dishes:\n" + _format_list(items, "item_name", "rating", "rating")
        elif intent == "popular":
            items = get_popular(_lazy_df(), top_n=8)
            reply = "Most popular right now:\n" + _format_list(items, "item_name", "score", "score")
        elif intent == "category_maybe":
            cat = _extract_category(request.new_message)
            if cat:
                items = find_by_category(cat, _lazy_df(), top_n=8)
                if items and "rating" in items[0]:
                    reply = f"Top items in {cat}:\n" + _format_list(items, "item_name", "rating", "rating")
                else:
                    reply = f"Popular items in {cat}:\n" + _format_list(items, "item_name", "score", "score")
            else:
                reply = "Please specify a category like Snacks, Beverage, Breakfast, Lunch, Pizza, Noodles, Thali, or Sandwich."
        elif intent == "recommend":
            rated = _top_rated(5)
            picks = [r["item_name"] for r in rated[:3]]
            reply = "You could try: " + ", ".join(picks) + ". Ask for 'highest rated' or 'popular' for more."
        else:
            menu = _menu_text()
            specials = _specials(2)
            pop = _pop_rank()
            rated = _top_rated(3)
            intro = "Hi! I’m your canteen assistant. I can suggest popular items, the highest rated dishes, or items by category.\n"
            reply = intro + "\nToday's specials: " + (", ".join(specials) if specials else "none") + "\nTop rated: " + ", ".join([r["item_name"] for r in rated]) + "\nMenu:\n" + menu + "\nPopularity ranks (top 10): " + str(pop)
        updated = request.history + [
            Content(role="user", parts=[Part(text=request.new_message)]),
            Content(role="model", parts=[Part(text=reply)])
        ]
        return ChatResponse(reply=reply, updated_history=updated)
    except Exception as e:
        raise HTTPException(500, f"{e}")

@app.get("/")
def root():
    return {"msg": "Chatbot Live ✅"}
