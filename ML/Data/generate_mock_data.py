# ml/data/generate_mock_data.py
import os
import csv
import random
from datetime import datetime, timedelta

OUT_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "mock_canteen_orders.csv")

items = [
    {"id": "D01", "name": "Samosa", "category": "Snack", "price": 20},
    {"id": "D02", "name": "Paneer Roll", "category": "Lunch", "price": 120},
    {"id": "D03", "name": "Tea", "category": "Beverage", "price": 15},
    {"id": "D04", "name": "Burger", "category": "Snack", "price": 60},
    {"id": "D05", "name": "Idli", "category": "Breakfast", "price": 50},
    {"id": "D06", "name": "Poha", "category": "Breakfast", "price": 40},
    {"id": "D07", "name": "Coffee", "category": "Beverage", "price": 25},
    {"id": "D08", "name": "Veg Thali", "category": "Lunch", "price": 90},
    {"id": "D09", "name": "Paratha", "category": "Breakfast", "price": 45},
    {"id": "D10", "name": "Shake", "category": "Beverage", "price": 70}
]

users = [f"U{i:03}" for i in range(1, 201)]  # 200 users
start_date = datetime(2025, 10, 1)
num_orders = 1200  # change as needed

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "order_id", "user_id", "item_id", "item_name",
        "category", "quantity", "total_price", "timestamp"
    ])
    writer.writeheader()
    for i in range(num_orders):
        user = random.choice(users)
        # weighted choice to make popularity realistic
        item = random.choices(items, weights=[10,15,20,9,8,7,6,11,5,4], k=1)[0]
        qty = random.choices([1,1,1,2,3], weights=[60,20,15,4,1], k=1)[0]
        date = start_date + timedelta(days=random.randint(0,34),
                                      hours=random.randint(6,21),
                                      minutes=random.randint(0,59))
        total_price = item["price"] * qty
        writer.writerow({
            "order_id": i + 1,
            "user_id": user,
            "item_id": item["id"],
            "item_name": item["name"],
            "category": item["category"],
            "quantity": qty,
            "total_price": total_price,
            "timestamp": date.strftime("%Y-%m-%d %H:%M:%S")
        })

print("Mock CSV created at:", OUT_CSV)
