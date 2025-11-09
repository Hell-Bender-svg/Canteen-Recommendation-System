from fastapi import FastAPI
from ML.API.recommend_api import router as recommend_router
from ML.chat_api_service import router as chat_router

app = FastAPI()

app.include_router(recommend_router, prefix="/api")
app.include_router(chat_router, prefix="/chat")

@app.get("/")
def root():
    return {"msg": "Backend Working"}
