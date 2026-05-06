# === FILE: cbf_model.py ===
# === PART OF: Movie Recommender System ===
# === IMPORTS FROM: data_loader (movies_df) ===

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import movies_df


# ============================================================
# Build TF-IDF matrix and cosine similarity matrix
# ============================================================

print("Building TF-IDF similarity matrix...")

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

tfidf_matrix      = tfidf.fit_transform(movies_df["features"])
similarity_matrix  = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Fast title lookup: lowercase title → dataframe index
title_to_index = pd.Series(
    movies_df.index,
    index=movies_df["title"].str.lower().str.strip()
)

print("TF-IDF matrix ready.\n")


# ============================================================
# Content-based recommendation function (original from v4)
# ============================================================

# Find movies similar to a given title using cosine similarity
def find_similar_movies(movie_name: str, n: int = 5,
                        filter_source: str = None):
    """
    Find movies similar to the one the user searched for.
    Uses cosine similarity on TF-IDF vectors (content-based).

    Returns:
        dict with 'searched_for' + 'recommendations'
        OR dict with 'error' or 'multiple_matches'
    """
    search = movie_name.strip().lower()

    # --- Match the title ---
    if search in title_to_index:
        idx           = title_to_index[search]
        matched_title = movies_df.loc[idx, "title"]
    else:
        partial = [t for t in title_to_index.index if search in t]
        if not partial:
            return {
                "error": f"No movie found for '{movie_name}'. "
                          "Try a different spelling or partial title."
            }
        if len(partial) == 1:
            idx           = title_to_index[partial[0]]
            matched_title = movies_df.loc[idx, "title"]
        else:
            options = [
                movies_df.loc[title_to_index[t], "title"]
                for t in partial[:8]
            ]
            return {
                "multiple_matches": options,
                "message": f"Found {len(partial)} matches — did you mean:"
            }

    # --- Get similarity scores ---
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:]  # skip the movie itself

    if filter_source:
        scores = [
            (i, s) for (i, s) in scores
            if movies_df.loc[i, "source"].lower() == filter_source.lower()
        ]

    top = scores[:n]

    results = []
    for rank, (movie_idx, score) in enumerate(top, start=1):
        row = movies_df.iloc[movie_idx]
        results.append({
            "rank":       rank,
            "title":      row["title"],
            "year":       int(row["year"]) if pd.notna(row["year"]) else "N/A",
            "genres":     row["genres"],
            "source":     row["source"],
            "similarity": round(score, 2)
        })

    return {
        "searched_for": matched_title,
        "recommendations": results
    }


# Helper: resolve a movie name to its dataframe index (used by hybrid)
def _resolve_title(movie_name):
    """Return (index, matched_title) or (None, error_dict)."""
    search = movie_name.strip().lower()
    if search in title_to_index:
        idx = title_to_index[search]
        return idx, movies_df.loc[idx, "title"]
    partial = [t for t in title_to_index.index if search in t]
    if not partial:
        return None, {"error": f"No movie found for '{movie_name}'."}
    if len(partial) == 1:
        idx = title_to_index[partial[0]]
        return idx, movies_df.loc[idx, "title"]
    options = [movies_df.loc[title_to_index[t], "title"] for t in partial[:8]]
    return None, {"multiple_matches": options,
                  "message": f"Found {len(partial)} matches — did you mean:"}
