import os
import random
import pandas as pd
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
from google import genai
from google.genai import types

from ML.API.recommend_api import (
    get_menu,
    get_popular,
    get_highest_rated,
    find_by_category,
    spicy_items
)

load_dotenv()
router = APIRouter(prefix="/chat", tags=["chat"])

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
    ai = genai.Client()
    MODEL = "gemini-2.5-flash"
except:
    ai = None

def rule_based(msg: str):
    t = msg.lower().strip()

    menu = get_menu()

    if any(g in t for g in ["hi", "hello", "hey"]):
        return random.choice([
            "Hey! What’s cooking today? 😄",
            "Hello! Ready to grab something yummy? 😋",
            "Hi! What can I help you choose today? 🍽️"
        ])

    if "price of" in t or "cost of" in t:
        name = t.replace("price of", "").replace("cost of", "").strip()
        for item in menu:
            if item["item_name"].lower() == name.lower():
                return f"{item['item_name']} costs ₹{item['price']}."
        return "That item is not on the menu."

    if "show menu" in t or "menu" in t:
        return "\n".join([f"- {i['item_name']} (₹{i['price']})" for i in menu])

    if "popular" in t:
        p = get_popular()
        return "\n".join([f"{i+1}. {x['item_name']}" for i, x in enumerate(p)])

    if "highest rated" in t or "best rated" in t:
        r = get_highest_rated()
        return "\n".join([f"{i+1}. {x['item_name']}" for i, x in enumerate(r)])

    if "spicy" in t:
        s = spicy_items()
        if not s:
            return "No spicy dishes found."
        return "\n".join([f"{i+1}. {x['item_name']}" for i, x in enumerate(s)])

    if "category" in t:
        cat = t.split("category")[-1].strip()
        c = find_by_category(cat)
        if not c:
            return f"No items found for category '{cat}'."
        return "\n".join([f"{i+1}. {x['item_name']}" for i, x in enumerate(c)])

    return None

def system_prompt():
    menu = get_menu()
    menu_text = "\n".join([f"- {i['item_name']} (₹{i['price']}) [{i['category']}]" for i in menu])

    return f"""
You are the official college canteen assistant.
You must never invent menu items or prices.

MENU:
{menu_text}

RULES:
- Only answer using the menu and user dataset facts.
- If asked about items not in the menu, say they are unavailable.
- You can speak casually and friendly.
"""

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    msg = request.new_message

    r = rule_based(msg)
    if r:
        updated = request.history + [
            Content(role="user", parts=[Part(text=msg)]),
            Content(role="model", parts=[Part(text=r)])
        ]
        return ChatResponse(reply=r, updated_history=updated)

    if not ai:
        raise HTTPException(503, "AI unavailable")

    prompt = system_prompt()

    convo = [types.Content(role="user", parts=[types.Part(text=prompt)])]

    for m in request.history:
        convo.append(types.Content(role=m.role, parts=[types.Part(text=p.text) for p in m.parts]))

    convo.append(types.Content(role="user", parts=[types.Part(text=msg)]))

    try:
        res = ai.models.generate_content(model=MODEL, contents=convo)
        reply = res.text
    except Exception as e:
        raise HTTPException(500, str(e))

    updated = request.history + [
        Content(role="user", parts=[Part(text=msg)]),
        Content(role="model", parts=[Part(text=reply)])
    ]
    return ChatResponse(reply=reply, updated_history=updated)
