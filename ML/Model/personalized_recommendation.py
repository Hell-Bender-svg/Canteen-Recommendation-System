# ML/Model/personalized_recommendation.py

import os
import pickle
import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset


class PersonalizedHybridRecommender:
    """
    Hybrid Recommender System:
    - Combines Collaborative Filtering (user-item interaction)
    - With Content-Based Filtering (user & item attributes)
    """

    def __init__(self, model_path="ML/Model/personalized_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.dataset = Dataset()
        self.user_features = None
        self.item_features = None

    def _prepare_lightfm_dataset(self, users_df, items_df, orders_df):
        """Prepares LightFM dataset structure"""
        users_df["_id"] = users_df["_id"].astype(str)
        items_df["_id"] = items_df["_id"].astype(str)
        orders_df["user_id"] = orders_df["user_id"].astype(str)
        orders_df["item_id"] = orders_df["item_id"].astype(str)

        # Define known features for LightFM
        self.dataset.fit(
            users=users_df["_id"].unique(),
            items=items_df["_id"].unique(),
            user_features=["age", "gender", "diet_preference", "fav_category"],
            item_features=["category"]
        )

        # Build interactions from orders (user_id, item_id, rating)
        interactions, _ = self.dataset.build_interactions([
            (u, i, float(r))
            for u, i, r in orders_df[["user_id", "item_id", "rating"]].values
        ])

        # Build user features
        user_features = self.dataset.build_user_features([
            (
                row["_id"],
                [
                    f"gender:{row.get('gender', 'unknown')}",
                    f"diet_preference:{row.get('diet_preference', 'none')}",
                    f"fav_category:{row.get('fav_category', 'none')}"
                ]
            )
            for _, row in users_df.iterrows()
        ])

        # Build item features
        item_features = self.dataset.build_item_features([
            (
                row["_id"],
                [f"category:{row.get('category', 'unknown')}"]
            )
            for _, row in items_df.iterrows()
        ])

        self.user_features = user_features
        self.item_features = item_features
        return interactions

    def train(self, users_df, items_df, orders_df):
        """Train the hybrid model"""
        interactions = self._prepare_lightfm_dataset(users_df, items_df, orders_df)

        model = LightFM(loss="warp", random_state=42)
        model.fit(
            interactions,
            user_features=self.user_features,
            item_features=self.item_features,
            epochs=25,
            num_threads=4
        )

        self.model = model

    def recommend_for_user(self, user_id, n=5):
        """Generate personalized recommendations for a user"""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded yet.")

        mapping = self.dataset.mapping()
        user_map, _, item_map, _ = mapping

        if user_id not in user_map:
            print(f"⚠️ User {user_id} not found in training data.")
            return []

        user_internal_id = user_map[user_id]
        n_items = len(item_map)

        scores = self.model.predict(
            user_internal_id,
            np.arange(n_items),
            user_features=self.user_features,
            item_features=self.item_features
        )

        top_items = np.argsort(-scores)[:n]
        reverse_item_map = {v: k for k, v in item_map.items()}
        return [reverse_item_map[i] for i in top_items]

    def save(self):
        """Save model and dataset"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "dataset": self.dataset,
                    "user_features": self.user_features,
                    "item_features": self.item_features
                },
                f
            )

    def load(self):
        """Load model and dataset"""
        with open(self.model_path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.dataset = data["dataset"]
        self.user_features = data["user_features"]
        self.item_features = data["item_features"]
