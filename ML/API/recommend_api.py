from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

app = FastAPI(title="Canteen Recommendation API", version="1.0.0")

# --- CORS (adjust for your frontend domain) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # replace "*" with your frontend origin in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data loading (repo-relative, safe) ---
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = (BASE_DIR / ".." / "Data" / "raw" / "mock_canteen_orders.csv").resolve()

def load_orders():
    df = pd.read_csv(DATA_PATH)
    # parse timestamp safely if present
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

# --- Core logic: popularity recommender ---
def get_popular(df: pd.DataFrame, top_n: int = 5, days: int | None = None):
    if days and "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        cutoff = pd.Timestamp.utcnow() - timedelta(days=days)
        df = df[df["timestamp"] >= cutoff]
    counts = df["item_name"].value_counts().reset_index()
    counts.columns = ["item_name", "order_count"]
    return counts.head(top_n).to_dict(orient="records")

@app.get("/", tags=["health"])
def root():
    return {"ok": True, "service": "canteen-recommendation", "version": "1.0.0"}

@app.get("/recommend", tags=["recommendations"])
def recommend(
    top_n: int = Query(5, ge=1, le=50),
    window_days: int | None = Query(None, ge=1, description="If set, use only recent N days"),
):
    """
    Returns top-N most-ordered dishes (optionally within a recent time window).
    Response format:
    {
      "top_n": 5,
      "window_days": 7,
      "recommendations": [{"item_name":"Tea","order_count":132}, ...]
    }
    """
    try:
        df = load_orders()
        if "item_name" not in df.columns:
            return {"error": "item_name column missing in dataset"}
        recs = get_popular(df, top_n=top_n, days=window_days)
        return {"top_n": top_n, "window_days": window_days, "recommendations": recs}
    except Exception as e:
        return {"error": str(e)}
