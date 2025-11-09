from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

DATASET_PATH = Path(__file__).resolve().parents[1] / "Data" / "raw" / "canteen_recommendation_dataset.csv"

def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "datetime" in df.columns:
        try:
            df["datetime"] = pd.to_datetime(df["datetime"])
        except Exception:
            pass
    return df

def load_orders() -> pd.DataFrame:
    return _read_csv(DATASET_PATH)

def get_menu(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if df is None:
        df = load_orders()
    cols = df.columns
    need = {"item_name", "price"}
    if not need.issubset(set(cols)):
        raise KeyError(f"Required columns missing for menu: {sorted(need)}; found: {list(cols)}")
    grouped = (
        df.groupby("item_name", as_index=False)
        .agg(
            price=("price", "median"),
            category=("category", "first"),
        )
    )
    grouped["price"] = grouped["price"].round(2)
    return grouped

def get_popular(df: Optional[pd.DataFrame] = None, top_n: int = 10) -> List[Dict]:
    if df is None:
        df = load_orders()
    cols = set(df.columns)
    if "item_name" not in cols:
        raise KeyError(f"item_name column missing. Columns found: {list(df.columns)}")
    if "purchase_count" in cols:
        score = df.groupby("item_name", as_index=False)["purchase_count"].sum().rename(columns={"purchase_count": "score"})
    elif "popularity_score" in cols:
        score = df.groupby("item_name", as_index=False)["popularity_score"].mean().rename(columns={"popularity_score": "score"})
    else:
        freq = df["item_name"].value_counts().reset_index()
        freq.columns = ["item_name", "score"]
        score = freq
    out = score.sort_values("score", ascending=False).head(top_n)
    return [{"item_name": r["item_name"], "score": float(r["score"])} for _, r in out.iterrows()]

def get_highest_rated(df: Optional[pd.DataFrame] = None, top_n: int = 10) -> List[Dict]:
    if df is None:
        df = load_orders()
    cols = set(df.columns)
    if not {"item_name", "rating"}.issubset(cols):
        raise KeyError(f"Required columns missing for highest rated: item_name, rating; found: {list(df.columns)}")
    rated = df.groupby("item_name", as_index=False)["rating"].mean()
    rated["rating"] = rated["rating"].round(2)
    out = rated.sort_values("rating", ascending=False).head(top_n)
    return [{"item_name": r["item_name"], "rating": float(r["rating"])} for _, r in out.iterrows()]

def find_by_category(category: str, df: Optional[pd.DataFrame] = None, top_n: int = 10) -> List[Dict]:
    if df is None:
        df = load_orders()
    mask = df["category"].str.lower() == category.lower()
    sub = df[mask]
    if sub.empty:
        return []
    if "rating" in sub.columns:
        rated = sub.groupby("item_name", as_index=False)["rating"].mean().sort_values("rating", ascending=False).head(top_n)
        return [{"item_name": r["item_name"], "rating": float(r["rating"])} for _, r in rated.iterrows()]
    freq = sub["item_name"].value_counts().reset_index().head(top_n)
    freq.columns = ["item_name", "score"]
    return [{"item_name": r["item_name"], "score": int(r["score"])} for _, r in freq.iterrows()]
