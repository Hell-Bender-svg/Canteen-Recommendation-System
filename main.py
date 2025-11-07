from fastapi import FastAPI
from ML.API.recommend_api import app as recommend_app
from ML.chat_api_service import app as chat_app

app = FastAPI(title="Canteen ML + Chat API")

app.mount("/recommend", recommend_app)
app.mount("/chat", chat_app)

@app.get("/")
def root():
    return {
        "message": "Canteen ML + Chat API is running!",
        "endpoints": [
            "/recommend",
            "/chat",
            "/docs"
        ]
    }
