from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ML.chat_api_service import router as chat_router

app = FastAPI(title="Canteen Chatbot + Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/chat")


@app.get("/")
def home():
    return {"ok": True, "service": "canteen-chatbot-api", "routes": ["/chat/chat"]}
