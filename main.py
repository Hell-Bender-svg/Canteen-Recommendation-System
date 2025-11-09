from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ML.API.recommend_api import router as recommend_router
from ML.chat_api_service import router as chat_router

app = FastAPI(
    title="Canteen Management System API",
    description="AI-powered chatbot and recommendation system for college canteen",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommend_router)
app.include_router(chat_router)

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Canteen Management System API",
        "endpoints": {
            "chat": "/chat/chat",
            "menu": "/recommend/menu",
            "popular": "/recommend/popular",
            "highest_rated": "/recommend/highest-rated",
            "spicy": "/recommend/spicy",
            "category": "/recommend/category/{category}",
            "search": "/recommend/search/{query}",
            "item": "/recommend/item/{item_name}"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "canteen-api"}