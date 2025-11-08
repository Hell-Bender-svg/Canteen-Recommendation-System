import pandas as pd
import os
fn = os.path.join(os.path.dirname(__file__), "raw", "mock_canteen_orders.csv")
df = pd.read_csv(fn, parse_dates=["timestamp"])

issues = []
if df.isnull().any().any():
    issues.append("Null values detected.")
if not df['order_id'].is_unique:
    issues.append("order_id not unique.")
if (df['total_price'] <= 0).any():
    issues.append("Non-positive total_price found.")
if issues:
    print("Validation issues:", issues)
else:
    print("Basic validation passed. Rows:", len(df))
