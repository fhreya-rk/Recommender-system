# === FILE: hybrid.py ===
# === PART OF: Movie Recommender System ===
# === IMPORTS FROM: data_loader (movies_df), cbf_model (similarity_matrix, _resolve_title), cf_model (svd_model, all_ml_user_ids, ml_trainset) ===

import pandas as pd

from data_loader import movies_df
from cbf_model import similarity_matrix, _resolve_title
from cf_model import svd_model, all_ml_user_ids, ml_trainset


# ============================================================
# Hybrid recommender: blends CF (SVD) + CBF (cosine similarity)
# Formula: final_score = 0.6 * cf_score + 0.4 * cbf_score
# ============================================================

# Combine collaborative and content-based scores into one ranking
def hybrid_recommend(user_id: str, movie_title: str, n: int = 5):
    """
    Combine CF (SVD) and CBF (cosine similarity) into one score.

    Parameters:
        user_id     : str  — MovieLens user ID (1–943)
        movie_title : str  — a movie title to anchor content similarity
        n           : int  — how many recommendations to return

    Returns:
        dict with 'searched_for', 'user_id', 'mode', 'recommendations'
        OR dict with 'error' / 'multiple_matches'
    """

    # --- Resolve the movie title to a dataframe index ---
    idx, resolved = _resolve_title(movie_title)
    if idx is None:
        return resolved  # error or multiple_matches dict
    matched_title = resolved

    # --- Check if user exists in MovieLens ---
    cf_available = user_id in all_ml_user_ids
    if not cf_available:
        print(f"  ⚠ User '{user_id}' not found in MovieLens. Using CBF only.")

    # --- Get CBF scores for all movies relative to the anchor ---
    cbf_scores_raw = list(enumerate(similarity_matrix[idx]))

    # --- Get CF scores: predict rating for every movie for this user ---
    all_ml_item_ids = [
        str(ml_trainset.to_raw_iid(iid)) for iid in ml_trainset.all_items()
    ]

    # --- Build combined scores ---
    candidates = []

    for movie_idx, cbf_score in cbf_scores_raw:
        if movie_idx == idx:
            continue  # skip the anchor movie itself

        cf_score = 0.0

        if cf_available:
            ml_item_id = all_ml_item_ids[movie_idx % len(all_ml_item_ids)]
            try:
                pred = svd_model.predict(user_id, ml_item_id)
                cf_score = pred.est
            except:
                cf_score = 0.0

        candidates.append((movie_idx, cf_score, cbf_score))

    # --- Normalize CF scores to 0–1 range ---
    if cf_available and candidates:
        cf_vals = [c[1] for c in candidates]
        cf_min  = min(cf_vals)
        cf_max  = max(cf_vals)
        cf_range = cf_max - cf_min if cf_max != cf_min else 1.0

        candidates = [
            (idx_c, (cf - cf_min) / cf_range, cbf)
            for (idx_c, cf, cbf) in candidates
        ]

    # --- Combine: 60% CF + 40% CBF (or 100% CBF if no CF data) ---
    scored = []
    for (movie_idx, cf_norm, cbf_norm) in candidates:
        if cf_available:
            final = 0.6 * cf_norm + 0.4 * cbf_norm
        else:
            final = cbf_norm

        scored.append((movie_idx, cf_norm, cbf_norm, final))

    # Sort by final score, highest first
    scored.sort(key=lambda x: x[3], reverse=True)
    top = scored[:n]

    # --- Build output ---
    results = []
    for rank, (movie_idx, cf_s, cbf_s, final_s) in enumerate(top, start=1):
        row = movies_df.iloc[movie_idx]
        results.append({
            "rank":        rank,
            "title":       row["title"],
            "year":        int(row["year"]) if pd.notna(row["year"]) else "N/A",
            "source":      row["source"],
            "genres":      row["genres"],
            "cf_score":    round(cf_s, 3),
            "cbf_score":   round(cbf_s, 3),
            "final_score": round(final_s, 3),
        })

    mode = "hybrid (CF+CBF)" if cf_available else "content-based only (CBF)"

    return {
        "searched_for":    matched_title,
        "user_id":         user_id,
        "mode":            mode,
        "recommendations": results
    }


# ============================================================
# Cold-start: recommend popular movies by genre for new users
# ============================================================

# Fall back to genre-based popularity ranking for brand-new users
def cold_start_recommend(genre_preference: str, n: int = 5):
    """
    Recommend popular movies for a new user based on genre preference.

    Parameters:
        genre_preference : str — e.g. "Action", "Romance", "Drama"
        n                : int — how many to return

    Returns:
        list of dicts with rank, title, year, source, genres, popularity
    """
    genre_lower = genre_preference.strip().lower()

    # Filter movies whose genre string contains the preference
    mask = movies_df["genres"].str.lower().str.contains(genre_lower, na=False)
    filtered = movies_df[mask].copy()

    if filtered.empty:
        return {"error": f"No movies found for genre '{genre_preference}'."}

    # Sort by vote_average (popularity), then vote_count as tiebreaker
    filtered = filtered.sort_values(
        by=["vote_average", "vote_count"],
        ascending=[False, False]
    ).head(n)

    results = []
    for rank, (_, row) in enumerate(filtered.iterrows(), start=1):
        results.append({
            "rank":       rank,
            "title":      row["title"],
            "year":       int(row["year"]) if pd.notna(row["year"]) else "N/A",
            "source":     row["source"],
            "genres":     row["genres"],
            "popularity": round(float(row["vote_average"]), 1),
        })

    return results
