# ML/Model/recommend_for_user.py
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parents[0]   # .../ML/Model
MODEL_PATH = BASE_DIR / "trained_model.pkl"

def recommend_for_user(user_id: str, top_n: int = 5):
    data = joblib.load(MODEL_PATH)
    pivot = data["pivot"]; item_sim = data["item_sim"]
    if user_id not in pivot.index:
        # Cold start → fall back to global popular
        popular = pivot.sum(axis=0).sort_values(ascending=False).head(top_n).index.tolist()
        return popular
    user_vector = pivot.loc[user_id]
    scores = (item_sim * user_vector).sum(axis=1).sort_values(ascending=False)
    already_bought = set(pivot.columns[user_vector > 0])
    recs = [i for i in scores.index if i not in already_bought][:top_n]
    return recs
