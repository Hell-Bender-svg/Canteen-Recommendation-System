import random
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List
from ML.API.recommend_api import (
    get_menu,
    get_popular,
    get_highest_rated,
    find_by_category,
)

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


def text_menu():
    items = get_menu()
    return "\n".join([f"- {i['item_name']} (₹{i['price']}) [{i['category']}]" for i in items])


def text_popular():
    items = get_popular(10)
    return "\n".join([f"{idx+1}. {i['item_name']} — score {i['score']}" for idx, i in enumerate(items)])


def text_rated():
    items = get_highest_rated(10)
    return "\n".join([f"{idx+1}. {i['item_name']} — rating {round(i['rating'], 2)}" for idx, i in enumerate(items)])


def spicy_items():
    menu = get_menu()
    spicy_words = ["spicy", "masala", "hot", "peri peri", "tandoori"]
    out = []
    for m in menu:
        name = m["item_name"].lower()
        if any(w in name for w in spicy_words):
            out.append(m)
    return out


def text_spicy():
    items = spicy_items()
    if not items:
        return "No spicy items found."
    return "\n".join([f"- {i['item_name']} (₹{i['price']})" for i in items])


def rule_engine(user_text: str):
    t = user_text.lower()

    greetings = ["hi", "hello", "hey", "hii", "yo"]
    if t in greetings or any(t.startswith(g) for g in greetings):
        return random.choice([
            "Hey! What’s cooking? 😊",
            "Hello! Craving something tasty? 😄",
            "Hi there! What would you like today? 🍽️",
        ])

    if "menu" in t:
        return f"Here is the menu:\n{text_menu()}"

    if "popular" in t or "trending" in t:
        return f"Top popular dishes:\n{text_popular()}"

    if "highest rated" in t or "high rating" in t or "best rated" in t:
        return f"Here are the highest rated dishes:\n{text_rated()}"

    if "spicy" in t or "hot" in t:
        return f"Here are some spicy options:\n{text_spicy()}"

    if "suggest" in t and "category" in t:
        cat = t.split("category")[-1].strip(" :,.")
        items = find_by_category(cat, 10)
        if not items:
            return f"No items found in category '{cat}'."
        lines = [f"{idx+1}. {i['item_name']} — score {i['score']}" for idx, i in enumerate(items)]
        return f"Here are some suggestions in {cat}:\n" + "\n".join(lines)

    if "breakfast" in t:
        return "Breakfast suggestions:\n" + "\n".join(
            [i['item_name'] for i in find_by_category("Breakfast", 10)]
        )

    if "lunch" in t:
        return "Lunch suggestions:\n" + "\n".join(
            [i['item_name'] for i in find_by_category("Lunch", 10)]
        )

    if "snack" in t or "snacks" in t:
        return "Snack ideas:\n" + "\n".join(
            [i['item_name'] for i in find_by_category("Snacks", 10)]
        )

    return (
        "I can help you with:\n"
        "- menu\n"
        "- popular items\n"
        "- highest rated dishes\n"
        "- spicy items\n"
        "- category suggestions\n"
        "Try: 'show menu', 'popular items', 'spicy food', or 'highest rated'"
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    reply = rule_engine(request.new_message)

    updated = request.history + [
        Content(role="user", parts=[Part(text=request.new_message)]),
        Content(role="model", parts=[Part(text=reply)]),
    ]

    return ChatResponse(reply=reply, updated_history=updated)


@router.get("/")
def test():
    return {"ok": True, "msg": "Chatbot active"}
