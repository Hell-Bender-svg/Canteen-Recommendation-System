from fastapi import FastAPI
from ML.API.recommend_api import router as recommend_router
from ML.chat_api_service import router as chat_router

app = FastAPI(title="Canteen ML + Chatbot API")

app.include_router(recommend_router)
app.include_router(chat_router)

@app.get("/")
def home():
    return {"ok": True, "msg": "Canteen API running"}
