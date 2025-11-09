import os
import re
import math
import random
import pandas as pd
from typing import List, Dict, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

MENU_PATH = os.getenv("MENU_PATH", "ML/Data/raw/menu.csv")
ORDERS_PATH = os.getenv("ORDERS_PATH", "ML/Data/raw/canteen_recommendation_dataset.csv")

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

def _load_menu() -> pd.DataFrame:
    df = pd.read_csv(MENU_PATH)
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    needed = {"item_name", "price"}
    if not needed.issubset(set(cols)):
        raise ValueError(f"menu.csv must have columns: {needed}")
    if "rating" not in df.columns:
        df["rating"] = pd.NA
    if "category" not in df.columns:
        df["category"] = pd.NA
    if "spicy_level" not in df.columns:
        df["spicy_level"] = 0
    return df

def _load_orders() -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(ORDERS_PATH)
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    except Exception:
        return None

MENU = _load_menu()
ORDERS = _load_orders()

def _menu_text(limit: Optional[int] = None) -> str:
    df = MENU.copy()
    if limit:
        df = df.head(limit)
    lines = []
    for _, r in df.iterrows():
        nm = str(r.get("item_name"))
        pr = r.get("price")
        cat = r.get("category")
        rt = r.get("rating")
        if pd.notna(rt):
            line = f"- {nm} — ₹{pr} ({cat}) | ★ {round(float(rt),2)}"
        else:
            line = f"- {nm} — ₹{pr} ({cat})"
        lines.append(line)
    return "\n".join(lines) if lines else "Menu is empty."

def _popular(top_n: int = 10) -> List[Dict]:
    if ORDERS is None or "item_name" not in ORDERS.columns:
        # Fallback: use purchase_count in menu if present, else random
        if "purchase_count" in MENU.columns:
            counts = MENU[["item_name", "purchase_count"]].dropna()
            counts = counts.sort_values("purchase_count", ascending=False).head(top_n)
            return [{"item_name": a, "score": float(b)} for a, b in zip(counts["item_name"], counts["purchase_count"])]
        sampled = MENU.sample(min(top_n, len(MENU))) if len(MENU) else MENU
        return [{"item_name": n, "score": 1.0} for n in sampled["item_name"].tolist()]
    vc = ORDERS["item_name"].value_counts().reset_index()
    vc.columns = ["item_name", "score"]
    vc = vc.head(top_n)
    return vc.to_dict(orient="records")

def _highest_rated(top_n: int = 10) -> List[Dict]:
    df = MENU.copy()
    if "rating" not in df.columns:
        return []
    df = df[pd.notna(df["rating"])]
    if df.empty:
        return []
    df = df.sort_values("rating", ascending=False).head(top_n)
    return [{"item_name": a, "rating": float(b)} for a, b in zip(df["item_name"], df["rating"])]

def _by_category(category: str, top_n: int = 10) -> List[Dict]:
    df = MENU.copy()
    if "category" not in df.columns:
        return []
    cat = category.strip().lower()
    df = df[df["category"].astype(str).str.lower().str.contains(re.escape(cat))]
    if df.empty:
        return []
    if "rating" in df.columns and df["rating"].notna().any():
        df = df.sort_values("rating", ascending=False)
    return [{"item_name": r["item_name"], "price": r["price"], "rating": (float(r["rating"]) if pd.notna(r["rating"]) else None)} for _, r in df.head(top_n).iterrows()]

def _spicy(top_n: int = 10) -> List[Dict]:
    df = MENU.copy()
    if "spicy_level" not in df.columns:
        return []
    df = df.sort_values("spicy_level", ascending=False)
    df = df[df["spicy_level"].fillna(0) > 0]
    return [{"item_name": r["item_name"], "spicy_level": float(r["spicy_level"]), "price": r["price"]} for _, r in df.head(top_n).iterrows()]

def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", s.lower()).strip()

def _fuzzy_find_item(query: str) -> Optional[Dict]:
    q = _normalize(query)
    if not q:
        return None
    best = None
    best_score = -1.0
    for _, r in MENU.iterrows():
        name = str(r["item_name"])
        n = _normalize(name)
        score = 0.0
        if n in q or q in n:
            score += 2.0
        nq = set(n.split())
        qq = set(q.split())
        inter = len(nq & qq)
        score += inter / (1 + len(qq))
        if score > best_score:
            best_score = score
            best = r
    return None if best is None or best_score < 0.15 else best.to_dict()

def _intent(message: str) -> str:
    t = _normalize(message)
    if any(t.startswith(g) or t == g for g in ["hi", "hello", "hey", "yo", "sup", "hola", "hii", "hiii"]):
        return "greet"
    if any(k in t for k in ["show menu", "menu", "list items", "see menu"]):
        return "menu"
    if any(k in t for k in ["popular", "trending", "most ordered"]):
        return "popular"
    if any(k in t for k in ["highest rated", "best rated", "high rating"]):
        return "highest_rated"
    if any(k in t for k in ["spicy", "spice"]):
        return "spicy"
    if "category" in t:
        return "category"
    if any(k in t for k in ["price", "cost", "how much", "rs", "₹"]):
        return "price"
    return "fallback"

def _reply_for_intent(intent: str, message: str) -> str:
    if intent == "greet":
        return random.choice([
            "hey! what’s cookin’? 😊",
            "hello! hungry for something tasty? 😄",
            "hi! what can i get you today? 🍽️",
            "hey there! feeling hungry? 😁",
        ])
    if intent == "menu":
        return f"here’s the menu:\n{_menu_text()}"
    if intent == "popular":
        recs = _popular(10)
        if not recs:
            return "no popularity data right now."
        lines = [f"{i+1}. {r['item_name']}" + (f" — score {round(float(r['score']),2)}" if 'score' in r and r['score'] is not None else "") for i, r in enumerate(recs)]
        return "top popular picks:\n" + "\n".join(lines)
    if intent == "highest_rated":
        recs = _highest_rated(10)
        if not recs:
            return "no rating data available."
        lines = [f"{i+1}. {r['item_name']} — ★ {round(float(r['rating']),2)}" for i, r in enumerate(recs)]
        return "highest rated dishes:\n" + "\n".join(lines)
    if intent == "spicy":
        recs = _spicy(10)
        if not recs:
            return "couldn’t find spicy items."
        lines = [f"{i+1}. {r['item_name']} — spicy level {int(r['spicy_level'])} — ₹{r['price']}" for i, r in enumerate(recs)]
        return "spicy picks:\n" + "\n".join(lines)
    if intent == "category":
        m = _normalize(message)
        part = m.split("category")[-1].strip()
        if not part:
            return "which category should i suggest from? (e.g., category snacks)"
        recs = _by_category(part, 10)
        if not recs:
            return f"no items found for category '{part}'."
        lines = []
        for i, r in enumerate(recs):
            rating = f" — ★ {round(float(r['rating']),2)}" if r.get("rating") is not None else ""
            lines.append(f"{i+1}. {r['item_name']} — ₹{r['price']}{rating}")
        return f"top in category '{part}':\n" + "\n".join(lines)
    if intent == "price":
        found = _fuzzy_find_item(message)
        if not found:
            return "tell me the item name, e.g., 'price of samosa'."
        nm = found.get("item_name")
        pr = found.get("price")
        return f"{nm} costs ₹{pr}."
    item = _fuzzy_find_item(message)
    if item:
        nm = item.get("item_name")
        pr = item.get("price")
        cat = item.get("category")
        rt = item.get("rating")
        rt_txt = f" | ★ {round(float(rt),2)}" if (rt is not None and not pd.isna(rt)) else ""
        return f"i think you meant '{nm}'. it’s ₹{pr} ({cat}){rt_txt}."
    return "i can show the menu, popular items, highest-rated dishes, spicy items, category-wise picks, or an item’s price. try: 'show menu', 'popular items', 'highest rated', 'spicy items', 'category snacks', or 'price of samosa'."

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    msg = request.new_message or ""
    intent = _intent(msg)
    reply = _reply_for_intent(intent, msg)
    updated = request.history + [
        Content(role="user", parts=[Part(text=request.new_message)]),
        Content(role="model", parts=[Part(text=reply)]),
    ]
    return ChatResponse(reply=reply, updated_history=updated)

@router.get("/")
def ping():
    return {"ok": True, "hint": "POST /chat/chat with {history, new_message}"}
