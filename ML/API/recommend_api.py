import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/recommend", tags=["recommend"])

DATA_PATH = "ML/Data/raw/canteen_recommendation_dataset.csv"
MENU_PATH = "ML/Data/raw/menu.csv"

def load_dataset():
    try:
        return pd.read_csv(DATA_PATH)
    except:
        raise HTTPException(500, "canteen_recommendation_dataset.csv missing")

def load_menu():
    try:
        return pd.read_csv(MENU_PATH)
    except:
        raise HTTPException(500, "menu.csv missing")

@router.get("/menu")
def get_menu():
    return load_menu().to_dict(orient="records")

@router.get("/popular")
def get_popular(top_n: int = 10):
    df = load_dataset()
    if "popularity_score" not in df.columns:
        raise HTTPException(500, "popularity_score missing")
    df = df.groupby("item_name")["popularity_score"].mean().sort_values(ascending=False)
    df = df.reset_index().head(top_n)
    return df.to_dict(orient="records")

@router.get("/highest-rated")
def get_highest_rated(top_n: int = 10):
    df = load_dataset()
    if "rating" not in df.columns:
        raise HTTPException(500, "rating missing")
    df = df.groupby("item_name")["rating"].mean().sort_values(ascending=False)
    df = df.reset_index().head(top_n)
    return df.to_dict(orient="records")

@router.get("/category/{cat}")
def find_by_category(cat: str, top_n: int = 10):
    df = load_dataset()
    df = df[df["category"].str.lower() == cat.lower()]
    if df.empty:
        return []
    df = df.groupby("item_name")["popularity_score"].mean().reset_index()
    df = df.sort_values("popularity_score", ascending=False).head(top_n)
    return df.to_dict(orient="records")

@router.get("/spicy")
def spicy_items():
    df = load_dataset()
    df = df[df["spicy_level"] >= 3]
    if df.empty:
        return []
    df = df.groupby("item_name")["spicy_level"].mean().sort_values(ascending=False)
    return df.reset_index().to_dict(orient="records")
