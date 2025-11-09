import os
import random
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

def get_canteen_data():
    try:
        menu = get_menu()
        popular = get_popular(10)
        highest_rated = get_highest_rated(10)
        spicy = spicy_items()
        
        return {
            "menu": menu,
            "popular": popular,
            "highest_rated": highest_rated,
            "spicy": spicy[:10] if spicy else []
        }
    except:
        return None

def create_system_instruction():
    data = get_canteen_data()
    if not data:
        return "You are a college canteen assistant. Help users with their queries about food and menu items."
    
    menu_items = []
    for item in data['menu']:
        menu_items.append(f"- {item['item_name']} (₹{item['price']}) [{item.get('category', 'General')}]")
    
    popular_items = []
    for idx, item in enumerate(data['popular'], 1):
        score = item.get('popularity_score', 0)
        popular_items.append(f"{idx}. {item['item_name']} (Popularity: {score})")
    
    rated_items = []
    for idx, item in enumerate(data['highest_rated'], 1):
        rating = item.get('rating', 0)
        rated_items.append(f"{idx}. {item['item_name']} (Rating: {rating:.1f}/5)")
    
    spicy_items_list = []
    for idx, item in enumerate(data['spicy'], 1):
        level = item.get('spicy_level', 0)
        spicy_items_list.append(f"{idx}. {item['item_name']} (Spice Level: {level})")
    
    return f"""You are the friendly college canteen chatbot assistant. Help students find food they'll love!

COMPLETE MENU:
{chr(10).join(menu_items)}

TOP 10 MOST POPULAR ITEMS:
{chr(10).join(popular_items)}

TOP 10 HIGHEST RATED ITEMS:
{chr(10).join(rated_items)}

SPICY ITEMS (Spice Level 3+):
{chr(10).join(spicy_items_list) if spicy_items_list else "No spicy items available"}

YOUR ROLE:
1. Be friendly, casual, and helpful - like talking to a friend
2. ONLY recommend items from the menu above - never invent items
3. When asked about prices, use exact prices from the menu
4. When asked for popular items, use the popularity list above
5. When asked for highest rated items, use the ratings list above
6. If someone asks about an item not on the menu, politely say it's unavailable
7. Consider meal times - suggest breakfast items in morning, lunch items midday, snacks in evening
8. You can suggest combos using available items
9. Be concise but friendly
10. Use emojis occasionally to be engaging

NEVER make up item names, prices, or ratings. Always use the data provided above."""

def quick_response(msg: str):
    t = msg.lower().strip()
    
    if len(t) < 20 and any(g in t for g in ["hi", "hello", "hey", "hii", "helo"]):
        return random.choice([
            "Hey! What's cooking today? 😄",
            "Hello! Ready to grab something yummy? 😋",
            "Hi there! What can I help you with? 🍽️",
            "Hey! Looking for something delicious? 🌟"
        ])
    
    if len(t) < 25 and any(g in t for g in ["bye", "goodbye", "see you", "later"]):
        return random.choice([
            "Goodbye! Enjoy your meal! 😊",
            "See you later! Come back soon! 👋",
            "Bye! Hope you find something tasty! 🍴"
        ])
    
    if len(t) < 25 and any(g in t for g in ["thanks", "thank you", "thx", "thanku"]):
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