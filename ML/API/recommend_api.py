import os
import pandas as pd
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

DATA_PATH = os.path.join("ML", "Data", "raw", "canteen_recommendation_dataset.csv")

router = APIRouter(prefix="/recommend", tags=["recommend"])

def _read_df() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=500, detail=f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    cols = [c.strip() for c in df.columns]
    df.columns = cols
    if "item_name" not in df.columns:
        raise HTTPException(status_code=500, detail=f"'item_name' column missing. Columns: {df.columns.tolist()}")
    if "category" not in df.columns:
        df["category"] = "General"
    if "price" not in df.columns:
        df["price"] = None
    return df

def load_orders() -> List[Dict[str, Any]]:
    return _read_df().to_dict(orient="records")

def get_menu() -> List[Dict[str, Any]]:
    df = _read_df()
    menu = (
        df[["item_name", "price", "category"]]
        .drop_duplicates()
        .sort_values("item_name")
        .reset_index(drop=True)
    )
    menu["available"] = True
    return menu.to_dict(orient="records")

def get_popular(top_n: int = 10) -> List[Dict[str, Any]]:
    df = _read_df()
    if "purchase_count" in df.columns:
        agg = df.groupby(["item_name", "category", "price"], dropna=False)["purchase_count"].sum().reset_index()
        agg = agg.sort_values("purchase_count", ascending=False)
        agg = agg.rename(columns={"purchase_count": "score"})
    elif "popularity_score" in df.columns:
        agg = df.groupby(["item_name", "category", "price"], dropna=False)["popularity_score"].mean().reset_index()
        agg = agg.sort_values("popularity_score", ascending=False)
        agg = agg.rename(columns={"popularity_score": "score"})
    else:
        agg = df["item_name"].value_counts().reset_index()
        agg.columns = ["item_name", "score"]
        agg = agg.merge(df[["item_name", "category", "price"]].drop_duplicates(), on="item_name", how="left")
    return agg.head(top_n).to_dict(orient="records")

def get_highest_rated(top_n: int = 10) -> List[Dict[str, Any]]:
    df = _read_df()
    if "rating" not in df.columns:
        raise HTTPException(status_code=400, detail="Dataset has no 'rating' column")
    agg = (
        df.groupby(["item_name", "category", "price"], dropna=False)["rating"]
        .mean()
        .reset_index()
        .sort_values("rating", ascending=False)
        .head(top_n)
    )
    return agg.to_dict(orient="records")

def find_by_category(category: str, top_n: int = 10) -> List[Dict[str, Any]]:
    df = _read_df()
    if "category" not in df.columns:
        raise HTTPException(status_code=400, detail="Dataset has no 'category' column")
    sub = df[df["category"].astype(str).str.lower() == category.lower()]
    if sub.empty:
        return []
    if "rating" in sub.columns:
        agg = sub.groupby(["item_name", "price"], dropna=False)["rating"].mean().reset_index()
        agg = agg.sort_values("rating", ascending=False).rename(columns={"rating": "score"})
    else:
        agg = sub["item_name"].value_counts().reset_index()
        agg.columns = ["item_name", "score"]
        agg = agg.merge(sub[["item_name", "price"]].drop_duplicates(), on="item_name", how="left")
    agg["category"] = category
    return agg.head(top_n).to_dict(orient="records")

@router.get("/menu")
def menu_endpoint():
    return {"menu": get_menu()}

@router.get("/popular")
def popular_endpoint(top_n: int = 10):
    return {"items": get_popular(top_n)}

@router.get("/highest-rated")
def highest_rated_endpoint(top_n: int = 10):
    return {"items": get_highest_rated(top_n)}

@router.get("/by-category")
def by_category_endpoint(category: str, top_n: int = 10):
    return {"category": category, "items": find_by_category(category, top_n)}
