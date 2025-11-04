# ml/data/process_for_model.py
import pandas as pd
import os

RAW = os.path.join(os.path.dirname(__file__), "raw", "mock_canteen_orders.csv")
OUT = os.path.join(os.path.dirname(__file__), "processed")
os.makedirs(OUT, exist_ok=True)
OUT_FN = os.path.join(OUT, "orders_clean.csv")

df = pd.read_csv(RAW, parse_dates=["timestamp"])
# examples of cleaning: drop duplicates, enforce types
df = df.drop_duplicates(subset=["order_id"])
df['quantity'] = df['quantity'].astype(int)
df.to_csv(OUT_FN, index=False)
print("Saved cleaned file to", OUT_FN)
