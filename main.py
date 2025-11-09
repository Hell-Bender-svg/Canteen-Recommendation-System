from fastapi import FastAPI
from ML.API import recommend_api
from ML.chat_api_service import app as chat_app

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Unified API running"}

app.mount("/recommend", recommend_api.app)
app.mount("/chat", chat_app)
