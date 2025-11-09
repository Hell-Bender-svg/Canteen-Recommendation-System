from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Optional
import pandas as pd
from datetime import timedelta

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = (BASE_DIR.parent.parent / "Data" / "raw" / "canteen_recommendation_dataset.csv").resolve()

def load_orders() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df

def get_popular(df: pd.DataFrame, top_n: int = 5):
    counts = df["item_name"].value_counts().reset_index()
    counts.columns = ["item_name", "order_count"]
    return counts.head(top_n).to_dict(orient="records")

def get_highest_rated(df: pd.DataFrame, top_n: int = 5):
    ranked = df.groupby("item_name")["rating"].mean().sort_values(ascending=False)
    return [{"item_name": i, "avg_rating": float(r)} for i, r in ranked.head(top_n).items()]

def find_by_category(df: pd.DataFrame, category: str):
    x = df[df["category"].str.lower() == category.lower()]
    if x.empty:
        return []
    return x["item_name"].unique().tolist()

@app.get("/")
def api_home():
    return {"ok": True}

@app.get("/popular")
def popular(top_n: int = 5):
    df = load_orders()
    return get_popular(df, top_n)

@app.get("/highest_rated")
def highest_rated(top_n: int = 5):
    df = load_orders()
    return get_highest_rated(df, top_n)

@app.get("/category")
def category(c: str):
    df = load_orders()
    return find_by_category(df, c)
