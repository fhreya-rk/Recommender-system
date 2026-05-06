# ============================================================
# Real-Time Movie Recommender — Content-Based Filtering
# User types a movie name → get similar movies instantly
# Based on: genres, keywords, similarity scoring
# ============================================================

import pandas as pd
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Step 1: Load movie data ---
print("Loading movie data...")

dataset_path = os.path.join(
    Path.home(), ".surprise_data", "ml-100k", "ml-100k", "u.item"
)

# Genre column names from the MovieLens dataset spec
genre_cols = [
    "unknown", "action", "adventure", "animation", "children",
    "comedy", "crime", "documentary", "drama", "fantasy",
    "noir", "horror", "musical", "mystery", "romance",
    "sci-fi", "thriller", "war", "western"
]

col_names = ["movie_id", "title", "release_date", "video_date", "imdb_url"] \
            + genre_cols

movies_df = pd.read_csv(
    dataset_path, sep="|", encoding="latin-1",
    header=None, names=col_names
)

# --- Step 2: Build a genre string for each movie ---
# Each movie has binary columns (1 = has genre, 0 = doesn't)
# We convert that into a text string like "action adventure thriller"
# so TF-IDF can process it

def build_genre_string(row):
    return " ".join([genre for genre in genre_cols if row[genre] == 1])

movies_df["genre_str"] = movies_df.apply(build_genre_string, axis=1)

# Drop movies with no genre info at all
movies_df = movies_df[movies_df["genre_str"].str.strip() != ""].reset_index(drop=True)

print(f"Loaded {len(movies_df)} movies.\n")


# --- Step 3: Build the TF-IDF matrix ---
# Converts genre strings into numerical vectors so we can
# mathematically compare any two movies

print("Building similarity matrix...")
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies_df["genre_str"])

# Cosine similarity: compares every movie to every other movie
# Result is a (n_movies x n_movies) matrix
# similarity[i][j] = how similar movie i is to movie j (0 to 1)
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Build a fast lookup: title (lowercase) → dataframe index
title_to_index = pd.Series(
    movies_df.index,
    index=movies_df["title"].str.lower()
)

print("Ready. Similarity matrix built.\n")


# --- Step 4: The core recommendation function ---

def find_similar_movies(movie_name: str, n: int = 5):
    """
    Given a movie name (partial or full), returns N similar movies.

    Parameters:
        movie_name : str — what the user typed
        n          : int — number of results to return (default 5)

    Returns:
        list of dicts  OR  dict with 'error' key
    """

    search = movie_name.strip().lower()

    # --- Search logic: exact match first, then partial match ---
    if search in title_to_index:
        idx = title_to_index[search]
        matched_title = movies_df.loc[idx, "title"]

    else:
        # Try partial match — find all titles containing the search term
        partial_matches = [
            title for title in title_to_index.index
            if search in title
        ]

        if not partial_matches:
            return {
                "error": f"No movie found matching '{movie_name}'. "
                         f"Try a different spelling."
            }

        if len(partial_matches) == 1:
            idx = title_to_index[partial_matches[0]]
            matched_title = movies_df.loc[idx, "title"]

        else:
            # Multiple matches — return them so user can pick
            options = [movies_df.loc[title_to_index[t], "title"]
                       for t in partial_matches[:8]]
            return {
                "multiple_matches": options,
                "message": f"Found {len(partial_matches)} movies. "
                           f"Did you mean one of these?"
            }

    # Get similarity scores for the matched movie vs all others
    scores = list(enumerate(similarity_matrix[idx]))

    # Sort by similarity score, highest first
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Skip index 0 (that's the movie itself, score = 1.0)
    scores = scores[1:n + 1]

    # Build result
    results = []
    for rank, (movie_idx, score) in enumerate(scores, start=1):
        row = movies_df.iloc[movie_idx]
        genres = row["genre_str"].replace(" ", ", ")
        results.append({
            "rank":       rank,
            "title":      row["title"],
            "genres":     genres,
            "similarity": round(score, 2)
        })

    return {
        "searched_for": matched_title,
        "recommendations": results
    }


# --- Step 5: Real-time search loop ---

print("=" * 55)
print(" Movie Recommender — search by movie name")
print(" Type part of a title (e.g. 'toy' or 'titanic')")
print(" Type 'quit' to exit")
print("=" * 55)

while True:
    user_input = input("\nSearch movie: ").strip()

    if user_input.lower() == "quit":
        print("Goodbye!")
        break

    if not user_input:
        print("  Please type a movie name.")
        continue

    result = find_similar_movies(user_input, n=5)

    # Case 1: Error (no match found)
    if "error" in result:
        print(f"\n  {result['error']}")

    # Case 2: Multiple matches — show options
    elif "multiple_matches" in result:
        print(f"\n  {result['message']}\n")
        for i, title in enumerate(result["multiple_matches"], 1):
            print(f"    {i}. {title}")
        print("\n  Re-type with more of the title to narrow it down.")

    # Case 3: Clean result
    else:
        print(f"\n  Because you searched: '{result['searched_for']}'")
        print(f"\n  {'Rank':<6} {'Title':<40} {'Genres':<30} {'Match'}")
        print("  " + "-" * 85)
        for rec in result["recommendations"]:
            print(
                f"  {rec['rank']:<6} "
                f"{rec['title']:<40} "
                f"{rec['genres']:<30} "
                f"{rec['similarity']}"
            )