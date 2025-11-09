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
        menu_items.append(f"- {item['item_name']} (₹{item['price']}) [{item.get('category', 'General')}] Rating: {item.get('rating', 'N/A')}/5")
    
    popular_items = []
    for idx, item in enumerate(data['popular'], 1):
        score = item.get('popularity_score', 0)
        popular_items.append(f"{idx}. {item['item_name']} (Popularity Score: {score:.1f})")
    
    rated_items = []
    for idx, item in enumerate(data['highest_rated'], 1):
        rating = item.get('rating', 0)
        rated_items.append(f"{idx}. {item['item_name']} (Rating: {rating:.1f}/5)")
    
    spicy_items_list = []
    for idx, item in enumerate(data['spicy'], 1):
        level = item.get('spicy_level', 0)
        spicy_items_list.append(f"{idx}. {item['item_name']} (Spice Level: {level:.1f})")
    
    return f"""You are the friendly college canteen chatbot assistant. Help students find food they'll love!

COMPLETE MENU (ALL ITEMS WITH PRICES AND RATINGS):
{chr(10).join(menu_items)}

TOP 10 MOST POPULAR ITEMS (BY POPULARITY SCORE):
{chr(10).join(popular_items)}

TOP 10 HIGHEST RATED ITEMS (BY CUSTOMER RATINGS):
{chr(10).join(rated_items)}

SPICY ITEMS (Spice Level 3 or higher):
{chr(10).join(spicy_items_list) if spicy_items_list else "No spicy items available currently"}

YOUR INSTRUCTIONS:
1. ALWAYS use the exact data provided above when answering questions
2. When asked for "highest rated" or "best rated" items, use the HIGHEST RATED list above
3. When asked for "popular" items, use the MOST POPULAR list above
4. When asked for "spicy" items, use the SPICY ITEMS list above
5. When asked about prices, use the exact prices from the COMPLETE MENU
6. NEVER invent or make up any items, prices, ratings, or information
7. If asked about an item not in the menu, say it's unavailable
8. Be friendly and conversational but always accurate
9. You can suggest meal combinations using only items from the menu
10. Consider meal times when suggesting items

RESPONSE FORMAT:
- Keep responses natural and friendly
- Use emojis occasionally
- Be concise but helpful
- Always base answers on the data above

Remember: Accuracy is critical. Only use information from the lists above."""

def quick_response(msg: str):
    t = msg.lower().strip()
    
    simple_greetings = ["hi", "hello", "hey"]
    if len(t) <= 10 and t in simple_greetings:
        return random.choice([
            "Hey! What can I help you with today? 😊",
            "Hello! Looking for something to eat? 🍽️",
            "Hi there! How can I assist you? 🌟"
        ])
    
    simple_goodbyes = ["bye", "goodbye", "thanks", "thank you"]
    if len(t) <= 15 and t in simple_goodbyes:
        return random.choice([
            "You're welcome! Enjoy your meal! 😊",
            "Happy to help! 👋",
            "Anytime! Have a great day! 🌟"
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
                temperature=0.3,
                max_output_tokens=800
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