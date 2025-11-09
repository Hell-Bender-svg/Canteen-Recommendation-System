import os
import random
import json
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
    MODEL = "gemini-2.0-flash-exp"
except:
    ai = None

def get_tools_data():
    try:
        menu = get_menu()
        popular = get_popular(5)
        highest_rated = get_highest_rated(5)
        spicy = spicy_items()[:5] if spicy_items() else []
        
        return {
            "menu": menu,
            "popular": popular,
            "highest_rated": highest_rated,
            "spicy": spicy
        }
    except:
        return None

def create_system_instruction():
    data = get_tools_data()
    if not data:
        return "You are a college canteen assistant. Help users with their queries."
    
    menu_text = "\n".join([
        f"- {i['item_name']} (₹{i['price']}) - {i.get('category', 'N/A')}" 
        for i in data['menu']
    ])
    
    popular_text = "\n".join([
        f"{idx+1}. {i['item_name']}" 
        for idx, i in enumerate(data['popular'])
    ])
    
    rated_text = "\n".join([
        f"{idx+1}. {i['item_name']}" 
        for idx, i in enumerate(data['highest_rated'])
    ])
    
    spicy_text = "\n".join([
        f"{idx+1}. {i['item_name']}" 
        for idx, i in enumerate(data['spicy'])
    ]) if data['spicy'] else "No spicy items available"
    
    return f"""You are the official college canteen chatbot assistant. Your job is to help students and staff with food recommendations, menu inquiries, and general canteen questions.

COMPLETE MENU:
{menu_text}

TOP 5 POPULAR ITEMS:
{popular_text}

TOP 5 HIGHEST RATED ITEMS:
{rated_text}

SPICY ITEMS:
{spicy_text}

INSTRUCTIONS:
1. Be friendly, casual, and helpful
2. Only recommend items from the menu above
3. When asked about prices, always use the exact prices shown
4. If asked about items not on the menu, politely say they are unavailable
5. When recommending food, consider the context (breakfast, lunch, dinner, snacks)
6. You can suggest combinations or meal ideas using available items
7. Be concise but informative
8. Use emojis occasionally to be friendly
9. If asked about categories, filter items by their category
10. When suggesting popular or highly rated items, use the data provided above

Remember: Never invent items, prices, or information not provided in the menu."""

def quick_response(msg: str):
    t = msg.lower().strip()
    
    if any(g in t for g in ["hi", "hello", "hey", "hii"]) and len(t) < 20:
        return random.choice([
            "Hey! What's cooking today? 😄",
            "Hello! Ready to grab something yummy? 😋",
            "Hi there! What can I help you with? 🍽️",
            "Hey! Looking for something delicious? 🌟"
        ])
    
    if any(g in t for g in ["bye", "goodbye", "see you"]) and len(t) < 25:
        return random.choice([
            "Goodbye! Enjoy your meal! 😊",
            "See you later! Come back soon! 👋",
            "Bye! Hope you find something tasty! 🍴"
        ])
    
    if any(g in t for g in ["thanks", "thank you", "thx"]) and len(t) < 25:
        return random.choice([
            "You're welcome! 😊",
            "Happy to help! 🌟",
            "Anytime! Enjoy! 😋"
        ])
    
    return None

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    msg = request.new_message.strip()
    
    if not msg:
        raise HTTPException(400, "Message cannot be empty")
    
    quick = quick_response(msg)
    if quick:
        updated = request.history + [
            Content(role="user", parts=[Part(text=msg)]),
            Content(role="model", parts=[Part(text=quick)])
        ]
        return ChatResponse(reply=quick, updated_history=updated)
    
    if not ai:
        raise HTTPException(503, "AI service unavailable. Please try again later.")
    
    try:
        system_instruction = create_system_instruction()
        
        convo = []
        for m in request.history:
            convo.append(types.Content(
                role=m.role, 
                parts=[types.Part(text=p.text) for p in m.parts]
            ))
        
        convo.append(types.Content(
            role="user", 
            parts=[types.Part(text=msg)]
        ))
        
        res = ai.models.generate_content(
            model=MODEL,
            contents=convo,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
                max_output_tokens=500
            )
        )
        
        reply = res.text.strip()
        
    except Exception as e:
        raise HTTPException(500, f"AI service error: {str(e)}")
    
    updated = request.history + [
        Content(role="user", parts=[Part(text=msg)]),
        Content(role="model", parts=[Part(text=reply)])
    ]
    
    return ChatResponse(reply=reply, updated_history=updated)