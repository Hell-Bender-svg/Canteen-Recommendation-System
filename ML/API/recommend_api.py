import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/recommend", tags=["recommend"])

DATA_PATH = "ML/Data/raw/canteen_recommendation_dataset.csv"
MENU_PATH = "ML/Data/raw/menu.csv"

def load_dataset():
    try:
        df = pd.read_csv(DATA_PATH)
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        raise HTTPException(404, "Recommendation dataset file not found")
    except Exception as e:
        raise HTTPException(500, f"Error loading dataset: {str(e)}")

def load_menu():
    try:
        df = pd.read_csv(MENU_PATH)
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        raise HTTPException(404, "Menu file not found")
    except Exception as e:
        raise HTTPException(500, f"Error loading menu: {str(e)}")

@router.get("/menu")
def get_menu():
    menu_df = load_menu()
    return menu_df.to_dict(orient="records")

@router.get("/popular")
def get_popular(top_n: int = 10):
    df = load_dataset()
    
    if "popularity_score" not in df.columns:
        raise HTTPException(400, "Dataset missing popularity_score column")
    
    if "item_name" not in df.columns:
        raise HTTPException(400, "Dataset missing item_name column")
    
    popular_df = df.groupby("item_name", as_index=False)["popularity_score"].mean()
    popular_df = popular_df.sort_values("popularity_score", ascending=False).head(top_n)
    
    return popular_df.to_dict(orient="records")

@router.get("/highest-rated")
def get_highest_rated(top_n: int = 10):
    df = load_dataset()
    
    if "rating" not in df.columns:
        raise HTTPException(400, "Dataset missing rating column")
    
    if "item_name" not in df.columns:
        raise HTTPException(400, "Dataset missing item_name column")
    
    rated_df = df.groupby("item_name", as_index=False)["rating"].mean()
    rated_df = rated_df.sort_values("rating", ascending=False).head(top_n)
    
    return rated_df.to_dict(orient="records")

@router.get("/category/{cat}")
def find_by_category(cat: str, top_n: int = 10):
    df = load_dataset()
    
    if "category" not in df.columns:
        raise HTTPException(400, "Dataset missing category column")
    
    filtered_df = df[df["category"].str.lower().str.strip() == cat.lower().strip()]
    
    if filtered_df.empty:
        return []
    
    if "popularity_score" in filtered_df.columns and "item_name" in filtered_df.columns:
        result_df = filtered_df.groupby("item_name", as_index=False)["popularity_score"].mean()
        result_df = result_df.sort_values("popularity_score", ascending=False).head(top_n)
    else:
        result_df = filtered_df[["item_name"]].drop_duplicates().head(top_n)
    
    return result_df.to_dict(orient="records")

@router.get("/spicy")
def spicy_items():
    df = load_dataset()
    
    if "spicy_level" not in df.columns:
        return []
    
    df["spicy_level_numeric"] = pd.to_numeric(df["spicy_level"], errors="coerce")
    spicy_df = df[df["spicy_level_numeric"] >= 3].copy()
    
    if spicy_df.empty:
        return []
    
    if "item_name" in spicy_df.columns:
        result_df = spicy_df.groupby("item_name", as_index=False)["spicy_level_numeric"].mean()
        result_df = result_df.rename(columns={"spicy_level_numeric": "spicy_level"})
        result_df = result_df.sort_values("spicy_level", ascending=False)
        return result_df.to_dict(orient="records")
    
    return []

@router.get("/search/{query}")
def search_items(query: str):
    menu_df = load_menu()
    
    if "item_name" not in menu_df.columns:
        raise HTTPException(400, "Menu missing item_name column")
    
    query_lower = query.lower().strip()
    matches = menu_df[menu_df["item_name"].str.lower().str.contains(query_lower, na=False)]
    
    return matches.to_dict(orient="records")

@router.get("/item/{item_name}")
def get_item_details(item_name: str):
    menu_df = load_menu()
    
    if "item_name" not in menu_df.columns:
        raise HTTPException(400, "Menu missing item_name column")
    
    item = menu_df[menu_df["item_name"].str.lower().str.strip() == item_name.lower().strip()]
    
    if item.empty:
        raise HTTPException(404, f"Item '{item_name}' not found in menu")
    
    return item.to_dict(orient="records")[0]