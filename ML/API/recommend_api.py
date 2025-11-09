from fastapi import APIRouter, Query
from pathlib import Path
import pandas as pd

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = (BASE_DIR / "Data" / "raw" / "canteen_recommendation_dataset.csv").resolve()

def load_orders():
    df = pd.read_csv(DATA_PATH)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    return df

def get_popular(df, top_n=5):
    counts = df["item_name"].value_counts().reset_index()
    counts.columns = ["item_name", "order_count"]
    return counts.head(top_n).to_dict(orient="records")

def get_highest_rated(df, top_n=5):
    g = df.groupby("item_name")["rating"].mean().reset_index()
    g = g.sort_values("rating", ascending=False)
    return g.head(top_n).to_dict(orient="records")

def find_by_category(df, category):
    x = df[df["category"].str.lower() == category.lower()]
    if x.empty:
        return []
    return x["item_name"].unique().tolist()

@router.get("/recommend/popular")
def recommend_popular(top_n: int = Query(5, ge=1, le=50)):
    df = load_orders()
    recs = get_popular(df, top_n)
    return {"mode": "popular", "items": recs}

@router.get("/recommend/top-rated")
def recommend_top_rated(top_n: int = Query(5, ge=1, le=50)):
    df = load_orders()
    recs = get_highest_rated(df, top_n)
    return {"mode": "top-rated", "items": recs}

@router.get("/recommend/category/{category}")
def recommend_by_category(category: str):
    df = load_orders()
    items = find_by_category(df, category)
    return {"mode": "category", "category": category, "items": items}
