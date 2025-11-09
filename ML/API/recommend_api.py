from fastapi import APIRouter, Query
import pandas as pd
from pathlib import Path

router = APIRouter(prefix="/recommend", tags=["recommend"])

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "Data" / "raw" / "canteen_recommendation_dataset.csv"
MENU_PATH = BASE_DIR / "Data" / "raw" / "menu.csv"


def load_dataset():
    return pd.read_csv(DATASET_PATH)


def load_menu():
    return pd.read_csv(MENU_PATH)


def get_menu():
    return load_menu().to_dict(orient="records")


def get_item_price(item):
    df = load_menu()
    r = df[df["item_name"].str.lower() == item.lower()]
    return None if r.empty else float(r.iloc[0]["price"])


def search_item(item):
    df = load_menu()
    r = df[df["item_name"].str.lower().str.contains(item.lower())]
    return r.to_dict(orient="records")


def find_by_category(category, limit=10):
    df = load_menu()
    r = df[df["category"].str.lower() == category.lower()]
    return r.head(limit).to_dict(orient="records")


def get_popular(limit=10):
    df = load_dataset()
    agg = df.groupby("item_name")["purchase_count"].sum().reset_index()
    agg = agg.sort_values("purchase_count", ascending=False).head(limit)
    out = agg.to_dict(orient="records")
    for o in out:
        o["score"] = int(o["purchase_count"])
    return out


def get_highest_rated(limit=10):
    df = load_dataset()
    agg = df.groupby("item_name")["rating"].mean().reset_index()
    agg = agg.sort_values("rating", ascending=False).head(limit)
    out = agg.to_dict(orient="records")
    for o in out:
        o["score"] = float(o["rating"])
    return out


@router.get("/menu")
def menu():
    return {"items": get_menu()}


@router.get("/price")
def price(item: str):
    p = get_item_price(item)
    return {"item": item, "price": p}


@router.get("/search")
def search(q: str):
    return {"results": search_item(q)}


@router.get("/category")
def category(cat: str, limit: int = 10):
    return {"category": cat, "items": find_by_category(cat, limit)}


@router.get("/popular")
def popular(limit: int = 10):
    return {"popular": get_popular(limit)}


@router.get("/highest_rated")
def highest_rated(limit: int = 10):
    return {"highest_rated": get_highest_rated(limit)}
