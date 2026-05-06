# === FILE: cf_model.py ===
# === PART OF: Movie Recommender System ===
# === IMPORTS FROM: surprise, joblib, os ===

import os
import joblib

from surprise import Dataset, SVD, KNNBasic
from surprise.model_selection import train_test_split as surprise_split


# ============================================================
# Model persistence path
# ============================================================

MODEL_PATH = "svd_model.pkl"


# ============================================================
# Load MovieLens 100K dataset and split into train/test
# ============================================================

print("Loading MovieLens 100K dataset...")
ml_data = Dataset.load_builtin("ml-100k")

# Split: 80% train, 20% test (fixed seed for reproducibility)
ml_trainset, ml_testset = surprise_split(ml_data, test_size=0.2, random_state=42)
print(f"  Training ratings: {ml_trainset.n_ratings}")
print(f"  Testing ratings:  {len(ml_testset)}\n")

# Build a set of all valid MovieLens user IDs (strings)
all_ml_user_ids = set(
    str(ml_trainset.to_raw_uid(uid)) for uid in ml_trainset.all_users()
)


# ============================================================
# Train (or load from cache) the SVD model
# ============================================================

if os.path.exists(MODEL_PATH):
    print(f"Loading cached SVD model from '{MODEL_PATH}'...")
    svd_model = joblib.load(MODEL_PATH)
    print("  SVD model loaded from cache.")
    svd_predictions = svd_model.test(ml_testset)
else:
    print("Training SVD model...")
    svd_model = SVD(random_state=42)
    svd_model.fit(ml_trainset)
    joblib.dump(svd_model, MODEL_PATH)
    print(f"  SVD model trained and saved to '{MODEL_PATH}'.")
    svd_predictions = svd_model.test(ml_testset)


# ============================================================
# Train the KNNBasic model (user-based collaborative filtering)
# ============================================================

print("Training KNNBasic model (user-based)...")
knn_model = KNNBasic(sim_options={"name": "cosine", "user_based": True}, verbose=False)
knn_model.fit(ml_trainset)
knn_predictions = knn_model.test(ml_testset)
print("  KNN model trained.\n")


# ============================================================
# CF recommendation function for a single user
# ============================================================

# Get top-N collaborative filtering recommendations for a user
def get_cf_recommendations(user_id: str, n: int = 5):
    """
    Return top-N movie recommendations for a MovieLens user using SVD.

    Parameters:
        user_id : str — MovieLens user ID (1–943)
        n       : int — how many recommendations to return

    Returns:
        list of dicts with movie_id and predicted_rating
        OR dict with 'error' key
    """
    if user_id not in all_ml_user_ids:
        return {"error": f"User '{user_id}' not found. Valid IDs: 1–943."}

    # Get all item IDs in the training set
    all_item_ids = [
        str(ml_trainset.to_raw_iid(iid)) for iid in ml_trainset.all_items()
    ]

    # Get items this user already rated
    inner_uid = ml_trainset.to_inner_uid(user_id)
    rated_ids = set(
        str(ml_trainset.to_raw_iid(iid))
        for (iid, _) in ml_trainset.ur[inner_uid]
    )

    # Predict ratings for unseen items
    unseen = [iid for iid in all_item_ids if iid not in rated_ids]
    preds = [(iid, svd_model.predict(user_id, iid).est) for iid in unseen]

    # Sort by predicted rating, take top N
    preds.sort(key=lambda x: x[1], reverse=True)
    top = preds[:n]

    results = []
    for rank, (movie_id, rating) in enumerate(top, start=1):
        results.append({
            "rank":             rank,
            "movie_id":         movie_id,
            "predicted_rating": round(rating, 2)
        })

    return results
