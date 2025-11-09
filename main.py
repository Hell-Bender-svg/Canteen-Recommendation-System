from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ML.API.recommend_api import router as recommend_router
from ML.chat_api_service import router as chat_router

app = FastAPI(title="Canteen Recommendation System", version="1.0.0")

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
def root():
    return {
        "message": "Canteen API is live",
        "endpoints": {
            "menu": "/recommend/menu",
            "popular": "/recommend/popular",
            "highest_rated": "/recommend/highest-rated",
            "by_category": "/recommend/by-category?category=Lunch",
            "chat_docs": "/docs",
            "chat_post": "/chat/chat"
        }
    }
