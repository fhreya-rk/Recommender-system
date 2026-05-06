# ============================================================
# Movie Recommender System (Beginner-Friendly)
# Uses the MovieLens 100K dataset and the Surprise library
# ============================================================

# --- Step 0: Import the libraries we need ---
# 'surprise' is a library built specifically for recommendation systems.
# 'pandas' helps us work with data in table format.
from surprise import Dataset, SVD, accuracy
from surprise.model_selection import train_test_split
import pandas as pd


# --- Step 1: Load the MovieLens 100K dataset ---
# This dataset contains 100,000 movie ratings from real users.
# Surprise can download it automatically the first time you run this.
print("Loading the MovieLens 100K dataset...")
data = Dataset.load_builtin("ml-100k")
print("Dataset loaded successfully!\n")


# --- Step 2: Split the data into training and testing sets ---
# We use 80% of the data to train the model and 20% to test it.
# Think of it like studying with 80% of a textbook and
# then taking a quiz on the remaining 20% to see how well you learned.
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
print(f"Training set size : {trainset.n_ratings} ratings")
print(f"Testing set size  : {len(testset)} ratings\n")


# --- Step 3: Train the recommendation model using SVD ---
# SVD (Singular Value Decomposition) is a popular algorithm that
# finds hidden patterns in how users rate movies.
# Don't worry about the math — Surprise handles it for you!
print("Training the SVD model (this may take a few seconds)...")
model = SVD(random_state=42)
model.fit(trainset)
print("Model trained successfully!\n")


# --- Step 4: Evaluate the model on the test set ---
# We check how close the model's predicted ratings are to the real ones.
# RMSE = Root Mean Squared Error (lower is better).
# MAE  = Mean Absolute Error   (lower is better).
print("Evaluating the model on the test set...")
predictions = model.test(testset)
print(f"  RMSE: {accuracy.rmse(predictions, verbose=False):.4f}")
print(f"  MAE : {accuracy.mae(predictions, verbose=False):.4f}")
print()


# --- Step 5: Predict a rating for a specific user and movie ---
# Let's predict what User "196" would rate Movie "302".
# 'est' is the estimated (predicted) rating.
sample_user = "196"
sample_movie = "302"
prediction = model.predict(sample_user, sample_movie)
print(f"Predicted rating for User {sample_user} on Movie {sample_movie}: "
      f"{prediction.est:.2f}  (scale: 1–5)\n")


# --- Step 6: Get the top 5 movie recommendations for one user ---
# Strategy:
#   1. Find all movies this user has NOT rated yet.
#   2. Predict a rating for each of those unseen movies.
#   3. Sort by predicted rating and pick the top 5.

target_user = "196"
print(f"Finding top 5 recommendations for User {target_user}...")

# Get the list of all movie IDs in the dataset
all_movie_ids = [str(iid) for iid in trainset.all_items()]
# Convert internal IDs back to the original movie IDs
all_movie_ids = [trainset.to_raw_iid(iid) for iid in trainset.all_items()]

# Get the movies this user has already rated
rated_by_user = [item for (item, _) in trainset.ur[trainset.to_inner_uid(target_user)]]
rated_movie_ids = [trainset.to_raw_iid(iid) for iid in rated_by_user]

# Find movies the user has NOT seen yet
unseen_movies = [mid for mid in all_movie_ids if mid not in rated_movie_ids]

# Predict a rating for every unseen movie
predicted_ratings = []
for movie_id in unseen_movies:
    pred = model.predict(target_user, movie_id)
    predicted_ratings.append((movie_id, pred.est))

# Sort by predicted rating (highest first) and take the top 5
predicted_ratings.sort(key=lambda x: x[1], reverse=True)
top_5 = predicted_ratings[:5]

# --- Step 7: Load movie titles so we can show names, not just IDs ---
# The MovieLens 100K item file maps movie IDs to their titles.
# Surprise stores the downloaded data in ~/.surprise_data/ml-100k/
import os
from pathlib import Path

dataset_path = os.path.join(Path.home(), ".surprise_data", "ml-100k", "ml-100k", "u.item")
movies_df = pd.read_csv(
    dataset_path,
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=["movie_id", "title"]
)
# Create a quick lookup dictionary: movie_id -> title
movie_titles = dict(zip(movies_df["movie_id"].astype(str), movies_df["title"]))

# --- Step 8: Print the top 5 recommendations with titles ---
print(f"\n{'Rank':<6} {'Movie Title':<45} {'Predicted Rating'}")
print("-" * 70)
for rank, (movie_id, rating) in enumerate(top_5, start=1):
    title = movie_titles.get(movie_id, f"Unknown (ID: {movie_id})")
    print(f"{rank:<6} {title:<45} {rating:.2f}")

print("\nDone! 🎬")
