# ============================================================
# Movie Recommender System — v2
# Adds: real user input + hyperparameter tuning with GridSearchCV
# ============================================================

from surprise import Dataset, SVD, accuracy
from surprise.model_selection import train_test_split, GridSearchCV
import pandas as pd
import os
from pathlib import Path


# --- Step 1: Load dataset ---
print("Loading MovieLens 100K dataset...")
data = Dataset.load_builtin("ml-100k")
print("Done.\n")


# --- Step 2: Tune the SVD model using GridSearchCV ---
# GridSearchCV tries every combination of the values below
# and finds which combo gives the lowest RMSE.
# This runs once — it takes ~1-2 minutes but is worth it.

print("Tuning model hyperparameters (takes ~1-2 mins)...")

param_grid = {
    "n_factors": [50, 100],       # number of hidden patterns to find
    "n_epochs":  [20, 30],        # how many learning passes
    "lr_all":    [0.005, 0.01],   # learning rate
    "reg_all":   [0.02, 0.1]      # regularisation (prevents overfitting)
}

gs = GridSearchCV(SVD, param_grid, measures=["rmse"], cv=3, n_jobs=-1)
gs.fit(data)

best_params = gs.best_params["rmse"]
print(f"Best hyperparameters found: {best_params}")
print(f"Best RMSE from tuning    : {gs.best_score['rmse']:.4f}\n")


# --- Step 3: Train final model using the best hyperparameters ---
print("Training final model with best settings...")
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

model = SVD(
    n_factors = best_params["n_factors"],
    n_epochs  = best_params["n_epochs"],
    lr_all    = best_params["lr_all"],
    reg_all   = best_params["reg_all"],
    random_state=42
)
model.fit(trainset)
print("Model trained.\n")


# --- Step 4: Evaluate on test set ---
predictions = model.test(testset)
print(f"Test RMSE : {accuracy.rmse(predictions, verbose=False):.4f}")
print(f"Test MAE  : {accuracy.mae(predictions, verbose=False):.4f}\n")


# --- Step 5: Load movie titles for display ---
dataset_path = os.path.join(
    Path.home(), ".surprise_data", "ml-100k", "ml-100k", "u.item"
)
movies_df = pd.read_csv(
    dataset_path, sep="|", encoding="latin-1",
    header=None, usecols=[0, 1], names=["movie_id", "title"]
)
movie_titles = dict(zip(movies_df["movie_id"].astype(str), movies_df["title"]))

# Build a set of all valid user IDs for validation
all_user_ids = set(
    str(trainset.to_raw_uid(uid)) for uid in trainset.all_users()
)


# --- Step 6: The recommendation function ---
# This is the core backend function.
# Give it any user_id string → get back top N recommendations.

def get_recommendations(user_id: str, n: int = 5):
    """
    Returns top-N movie recommendations for a given user.

    Parameters:
        user_id : str  — the user ID (must exist in the dataset)
        n       : int  — how many recommendations to return (default 5)

    Returns:
        list of dicts with 'rank', 'title', 'predicted_rating'
        OR a dict with 'error' if the user ID is invalid
    """

    # Validate: does this user exist?
    if user_id not in all_user_ids:
        return {
            "error": f"User '{user_id}' not found. "
                     f"Valid IDs are between 1 and 943."
        }

    # Get all movie IDs in the dataset
    all_movie_ids = [
        trainset.to_raw_iid(iid) for iid in trainset.all_items()
    ]

    # Get movies this user has already rated
    inner_uid = trainset.to_inner_uid(user_id)
    rated_ids = [
        trainset.to_raw_iid(iid)
        for (iid, _) in trainset.ur[inner_uid]
    ]

    # Predict ratings for every movie the user hasn't seen
    unseen = [mid for mid in all_movie_ids if mid not in rated_ids]
    preds  = [(mid, model.predict(user_id, mid).est) for mid in unseen]

    # Sort highest predicted rating first, take top N
    preds.sort(key=lambda x: x[1], reverse=True)
    top_n = preds[:n]

    # Build clean output
    results = []
    for rank, (movie_id, rating) in enumerate(top_n, start=1):
        results.append({
            "rank":             rank,
            "title":            movie_titles.get(movie_id, f"ID:{movie_id}"),
            "predicted_rating": round(rating, 2)
        })

    return results


# --- Step 7: Accept real user input from the terminal ---
# This loop keeps asking for a user ID until the user types 'quit'.

print("=" * 55)
print(" Movie Recommender — type a user ID to get suggestions")
print(" Valid user IDs: 1 to 943  |  type 'quit' to exit")
print("=" * 55)

while True:
    user_input = input("\nEnter user ID: ").strip()

    if user_input.lower() == "quit":
        print("Goodbye!")
        break

    results = get_recommendations(user_id=user_input, n=5)

    # Handle error case (invalid user)
    if isinstance(results, dict) and "error" in results:
        print(f"  Error: {results['error']}")
        continue

    # Print recommendations
    print(f"\n  Top 5 recommendations for User {user_input}:\n")
    print(f"  {'Rank':<6} {'Movie Title':<45} {'Predicted Rating'}")
    print("  " + "-" * 62)
    for rec in results:
        print(f"  {rec['rank']:<6} {rec['title']:<45} {rec['predicted_rating']}")