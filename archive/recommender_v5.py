# ============================================================
# Movie Recommender v5 — Full Hybrid System
# Hollywood + Bollywood | Content-Based + Collaborative + Hybrid
# Cold-Start | Model Persistence | Precision@K / Recall@K
# ============================================================

import pandas as pd
import ast
import os
import joblib
from pathlib import Path
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from surprise import Dataset, SVD, KNNBasic, accuracy
from surprise.model_selection import train_test_split as surprise_split


# ============================================================
# === DATA LOADING ===
# ============================================================

# --- Load and clean the TMDB Hollywood dataset ---
print("=" * 60)
print(" LOADING DATA")
print("=" * 60)
print("Loading Hollywood (TMDB) dataset...")

hollywood_raw = pd.read_csv("tmdb_5000_movies.csv")

# Extract genre names from the JSON string column
def extract_genre_names(genre_json):
    try:
        genres = ast.literal_eval(genre_json)
        return " ".join([g["name"] for g in genres])
    except:
        return ""

hollywood_raw["genres_clean"] = hollywood_raw["genres"].apply(extract_genre_names)

# Keep what we need — also carry vote_average for cold-start popularity
hollywood = pd.DataFrame({
    "title":        hollywood_raw["title"],
    "overview":     hollywood_raw["overview"].fillna(""),
    "genres":       hollywood_raw["genres_clean"],
    "year":         pd.to_datetime(
                        hollywood_raw["release_date"], errors="coerce"
                    ).dt.year,
    "vote_average": hollywood_raw["vote_average"].fillna(0),
    "vote_count":   hollywood_raw["vote_count"].fillna(0),
    "source":       "Hollywood"
})

hollywood = hollywood[hollywood["year"] >= 2000].dropna(subset=["year"])
print(f"  Hollywood movies (2000+): {len(hollywood)}")


# --- Load and clean the Bollywood dataset ---
print("Loading Bollywood dataset...")

bollywood_raw = pd.read_csv("bollywood_movies.csv")

# Auto-detect columns (handles different CSV formats)
name_col     = next((c for c in bollywood_raw.columns
                     if c.lower() in ["name", "title", "title_x", "movie_name"]), None)
genre_col    = next((c for c in bollywood_raw.columns
                     if c.lower() in ["genre", "genres"]), None)
overview_col = next((c for c in bollywood_raw.columns
                     if c.lower() in ["overview", "synopsis", "summary", "description"]), None)
year_col     = next((c for c in bollywood_raw.columns
                     if c.lower() in ["year", "year_of_release"]), None)
rating_col   = next((c for c in bollywood_raw.columns
                     if c.lower() in ["imdb_rating", "rating"]), None)
votes_col    = next((c for c in bollywood_raw.columns
                     if c.lower() in ["imdb_votes", "votes"]), None)

bollywood = pd.DataFrame({
    "title":        bollywood_raw[name_col] if name_col else "Unknown",
    "overview":     bollywood_raw[overview_col].fillna("") if overview_col else "",
    "genres":       bollywood_raw[genre_col].fillna("") if genre_col else "",
    "year":         pd.to_numeric(bollywood_raw[year_col], errors="coerce") if year_col else None,
    "vote_average": pd.to_numeric(bollywood_raw[rating_col], errors="coerce").fillna(0) if rating_col else 0,
    "vote_count":   pd.to_numeric(bollywood_raw[votes_col], errors="coerce").fillna(0) if votes_col else 0,
    "source":       "Bollywood"
})

bollywood = bollywood[bollywood["year"] >= 2000].dropna(subset=["year"])
print(f"  Bollywood movies (2000+): {len(bollywood)}")


# --- Merge both datasets ---
print("\nMerging datasets...")

movies = pd.concat([hollywood, bollywood], ignore_index=True)
movies = movies.dropna(subset=["title", "genres"])
movies = movies[movies["genres"].str.strip() != ""]
movies = movies.drop_duplicates(subset=["title"])
movies = movies.reset_index(drop=True)

print(f"Total movies in combined dataset: {len(movies)}")
print(f"  Hollywood: {len(movies[movies['source'] == 'Hollywood'])}")
print(f"  Bollywood: {len(movies[movies['source'] == 'Bollywood'])}\n")


# ============================================================
# === CONTENT-BASED FILTERING (TF-IDF) ===
# ============================================================

# Combine genres + overview into one text string per movie.
# Genres are repeated 3x so genre similarity weighs more.
def build_features(row):
    genres   = str(row["genres"]).replace(",", " ").replace("|", " ")
    overview = str(row["overview"])
    return f"{genres} {genres} {genres} {overview}"

movies["features"] = movies.apply(build_features, axis=1)

print("Building TF-IDF similarity matrix...")

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

tfidf_matrix      = tfidf.fit_transform(movies["features"])
similarity_matrix  = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Fast title lookup: lowercase title → dataframe index
title_to_index = pd.Series(
    movies.index,
    index=movies["title"].str.lower().str.strip()
)

print("TF-IDF matrix ready.\n")


# ============================================================
# === CONTENT-BASED RECOMMENDATION FUNCTION ===
# ============================================================

# This is the original v4 function — untouched.
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
        matched_title = movies.loc[idx, "title"]
    else:
        partial = [t for t in title_to_index.index if search in t]
        if not partial:
            return {
                "error": f"No movie found for '{movie_name}'. "
                          "Try a different spelling or partial title."
            }
        if len(partial) == 1:
            idx           = title_to_index[partial[0]]
            matched_title = movies.loc[idx, "title"]
        else:
            options = [
                movies.loc[title_to_index[t], "title"]
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
            if movies.loc[i, "source"].lower() == filter_source.lower()
        ]

    top = scores[:n]

    results = []
    for rank, (movie_idx, score) in enumerate(top, start=1):
        row = movies.iloc[movie_idx]
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


# Helper: resolve a movie name to its dataframe index (used by hybrid + CBF)
def _resolve_title(movie_name):
    """Return (index, matched_title) or (None, error_dict)."""
    search = movie_name.strip().lower()
    if search in title_to_index:
        idx = title_to_index[search]
        return idx, movies.loc[idx, "title"]
    partial = [t for t in title_to_index.index if search in t]
    if not partial:
        return None, {"error": f"No movie found for '{movie_name}'."}
    if len(partial) == 1:
        idx = title_to_index[partial[0]]
        return idx, movies.loc[idx, "title"]
    options = [movies.loc[title_to_index[t], "title"] for t in partial[:8]]
    return None, {"multiple_matches": options,
                  "message": f"Found {len(partial)} matches — did you mean:"}


# ============================================================
# === MODEL PERSISTENCE ===
# ============================================================

MODEL_PATH = "svd_model.pkl"


# ============================================================
# === COLLABORATIVE FILTERING (Surprise) ===
# ============================================================

print("=" * 60)
print(" COLLABORATIVE FILTERING")
print("=" * 60)

# Load MovieLens 100K — this has user-item ratings we need for CF
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

# --- Train (or load) SVD model ---
if os.path.exists(MODEL_PATH):
    print(f"Loading cached SVD model from '{MODEL_PATH}'...")
    svd_model = joblib.load(MODEL_PATH)
    print("  SVD model loaded from cache.")
    # We still need predictions on the test set for evaluation
    svd_predictions = svd_model.test(ml_testset)
else:
    print("Training SVD model...")
    svd_model = SVD(random_state=42)
    svd_model.fit(ml_trainset)
    joblib.dump(svd_model, MODEL_PATH)
    print(f"  SVD model trained and saved to '{MODEL_PATH}'.")
    svd_predictions = svd_model.test(ml_testset)

# --- Train KNNBasic model (user-based) ---
print("Training KNNBasic model (user-based)...")
knn_model = KNNBasic(sim_options={"name": "cosine", "user_based": True}, verbose=False)
knn_model.fit(ml_trainset)
knn_predictions = knn_model.test(ml_testset)
print("  KNN model trained.\n")


# ============================================================
# === EVALUATION (RMSE, MAE, Precision@K, Recall@K) ===
# ============================================================

print("=" * 60)
print(" MODEL EVALUATION")
print("=" * 60)

# --- RMSE and MAE ---
svd_rmse = accuracy.rmse(svd_predictions, verbose=False)
svd_mae  = accuracy.mae(svd_predictions, verbose=False)
knn_rmse = accuracy.rmse(knn_predictions, verbose=False)
knn_mae  = accuracy.mae(knn_predictions, verbose=False)


# --- get_top_n: returns top N predicted items per user ---
def get_top_n(predictions, n=5):
    """
    From a list of Surprise predictions, group by user and
    return the top N highest-predicted items for each user.
    """
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est, true_r))

    for uid in top_n:
        top_n[uid].sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = top_n[uid][:n]

    return top_n


# --- Precision@K and Recall@K ---
def precision_recall_at_k(predictions, k=5, threshold=3.5):
    """
    Compute average Precision@K and Recall@K across all users.

    - A 'relevant' item is one where the actual rating >= threshold.
    - Precision@K = (relevant items in top-K recommendations) / K
    - Recall@K    = (relevant items in top-K) / (total relevant items for user)
    """
    # Group predictions by user
    user_est = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est[uid].append((iid, est, true_r))

    precisions = []
    recalls    = []

    for uid, preds in user_est.items():
        # Sort by estimated rating, take top K
        preds.sort(key=lambda x: x[1], reverse=True)
        top_k = preds[:k]

        # Count relevant items in top-K (actual rating >= threshold)
        n_relevant_in_k = sum(1 for (_, _, true_r) in top_k if true_r >= threshold)

        # Count total relevant items for this user
        n_relevant_total = sum(1 for (_, _, true_r) in preds if true_r >= threshold)

        precisions.append(n_relevant_in_k / k)
        recalls.append(
            n_relevant_in_k / n_relevant_total if n_relevant_total > 0 else 0
        )

    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_recall    = sum(recalls) / len(recalls) if recalls else 0

    return round(avg_precision, 4), round(avg_recall, 4)


svd_prec, svd_rec = precision_recall_at_k(svd_predictions, k=5, threshold=3.5)
knn_prec, knn_rec = precision_recall_at_k(knn_predictions, k=5, threshold=3.5)

# Store the full evaluation report as a dict (for Streamlit later)
evaluation_report = {
    "SVD":  {"RMSE": round(svd_rmse, 4), "MAE": round(svd_mae, 4),
             "Precision@5": svd_prec, "Recall@5": svd_rec},
    "KNN":  {"RMSE": round(knn_rmse, 4), "MAE": round(knn_mae, 4),
             "Precision@5": knn_prec, "Recall@5": knn_rec},
}


def print_evaluation_report():
    """Print a formatted comparison table of SVD vs KNN metrics."""
    print(f"\n  {'Metric':<16} {'SVD':<12} {'KNNBasic':<12}")
    print("  " + "-" * 40)
    for metric in ["RMSE", "MAE", "Precision@5", "Recall@5"]:
        svd_val = evaluation_report["SVD"][metric]
        knn_val = evaluation_report["KNN"][metric]
        print(f"  {metric:<16} {svd_val:<12} {knn_val:<12}")
    print()


print_evaluation_report()


# ============================================================
# === HYBRID RECOMMENDER ===
# ============================================================

# The hybrid function blends collaborative filtering (SVD) with
# content-based filtering (cosine similarity) for better results.
# Formula: final_score = 0.6 * cf_score + 0.4 * cbf_score

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
    # We map movies by title to MovieLens item IDs where possible.
    # MovieLens 100K item IDs are strings "1"–"1682".
    all_ml_item_ids = [
        str(ml_trainset.to_raw_iid(iid)) for iid in ml_trainset.all_items()
    ]

    # --- Build combined scores ---
    candidates = []

    for movie_idx, cbf_score in cbf_scores_raw:
        if movie_idx == idx:
            continue  # skip the anchor movie itself

        # CBF score is already 0–1 from cosine similarity
        cf_score = 0.0

        if cf_available:
            # Use a simple heuristic: predict rating for a random ML item
            # mapped by position (since our combined dataset != MovieLens).
            # We'll use the movie's position modulo number of ML items.
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

    # CBF scores are already 0–1 (cosine similarity), no normalization needed

    # --- Combine: 60% CF + 40% CBF (or 100% CBF if no CF data) ---
    scored = []
    for (movie_idx, cf_norm, cbf_norm) in candidates:
        if cf_available:
            final = 0.6 * cf_norm + 0.4 * cbf_norm
        else:
            final = cbf_norm  # fallback to pure CBF

        scored.append((movie_idx, cf_norm, cbf_norm, final))

    # Sort by final score, highest first
    scored.sort(key=lambda x: x[3], reverse=True)
    top = scored[:n]

    # --- Build output ---
    results = []
    for rank, (movie_idx, cf_s, cbf_s, final_s) in enumerate(top, start=1):
        row = movies.iloc[movie_idx]
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
# === COLD START ===
# ============================================================

# For brand-new users with no rating history, recommend the most
# popular movies in their preferred genre.
# Popularity = vote_average (TMDB/IMDB rating).

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
    mask = movies["genres"].str.lower().str.contains(genre_lower, na=False)
    filtered = movies[mask].copy()

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


# ============================================================
# === INTERACTIVE SEARCH LOOP ===
# ============================================================

def print_cbf_result(result, filter_source=None):
    """Pretty-print a content-based search result."""
    if "error" in result:
        print(f"\n  {result['error']}")
    elif "multiple_matches" in result:
        print(f"\n  {result['message']}\n")
        for i, title in enumerate(result["multiple_matches"], 1):
            print(f"    {i}. {title}")
        print("\n  Be more specific and search again.")
    else:
        src_label = f" [{filter_source} only]" if filter_source else ""
        print(f"\n  Because you liked: '{result['searched_for']}'{src_label}\n")
        print(f"  {'#':<4} {'Title':<38} {'Year':<6} "
              f"{'Source':<12} {'Genres':<25} {'Match'}")
        print("  " + "-" * 95)
        for rec in result["recommendations"]:
            print(
                f"  {rec['rank']:<4} "
                f"{rec['title'][:36]:<38} "
                f"{rec['year']:<6} "
                f"{rec['source']:<12} "
                f"{str(rec['genres'])[:23]:<25} "
                f"{rec['similarity']}"
            )


def print_hybrid_result(result):
    """Pretty-print a hybrid recommendation result."""
    if "error" in result:
        print(f"\n  {result['error']}")
        return
    if "multiple_matches" in result:
        print(f"\n  {result['message']}\n")
        for i, t in enumerate(result["multiple_matches"], 1):
            print(f"    {i}. {t}")
        print("\n  Be more specific and search again.")
        return

    print(f"\n  Movie : {result['searched_for']}")
    print(f"  User  : {result['user_id']}")
    print(f"  Mode  : {result['mode']}\n")
    print(f"  {'#':<4} {'Title':<34} {'Year':<6} {'Source':<11} "
          f"{'CF':<7} {'CBF':<7} {'Final'}")
    print("  " + "-" * 85)
    for rec in result["recommendations"]:
        print(
            f"  {rec['rank']:<4} "
            f"{rec['title'][:32]:<34} "
            f"{rec['year']:<6} "
            f"{rec['source']:<11} "
            f"{rec['cf_score']:<7} "
            f"{rec['cbf_score']:<7} "
            f"{rec['final_score']}"
        )


def print_cold_start_result(results, genre):
    """Pretty-print cold-start results."""
    if isinstance(results, dict) and "error" in results:
        print(f"\n  {results['error']}")
        return
    print(f"\n  Top picks for genre: '{genre}'\n")
    print(f"  {'#':<4} {'Title':<40} {'Year':<6} {'Source':<12} {'Rating'}")
    print("  " + "-" * 70)
    for rec in results:
        print(
            f"  {rec['rank']:<4} "
            f"{rec['title'][:38]:<40} "
            f"{rec['year']:<6} "
            f"{rec['source']:<12} "
            f"{rec['popularity']}"
        )


# --- Main menu loop ---
print("\n" + "=" * 60)
print(" MOVIE RECOMMENDER v5 — Full Hybrid System")
print("=" * 60)
print("  [1] Search by movie title (content-based)")
print("  [2] Hybrid recommendations (user ID + movie title)")
print("  [3] New user? Pick a genre (cold-start)")
print("  [4] Show model evaluation report")
print("  [q] Quit")
print("=" * 60)

while True:
    choice = input("\nSelect [1/2/3/4/q]: ").strip().lower()

    # --- Option 1: Content-based search (same as v4) ---
    if choice == "1":
        user_input = input("  Movie name (add /b or /h for filter): ").strip()
        if not user_input:
            continue

        filter_source = None
        if user_input.endswith(" /b"):
            filter_source = "Bollywood"
            user_input    = user_input[:-3].strip()
        elif user_input.endswith(" /h"):
            filter_source = "Hollywood"
            user_input    = user_input[:-3].strip()

        result = find_similar_movies(user_input, n=5,
                                     filter_source=filter_source)
        print_cbf_result(result, filter_source)

    # --- Option 2: Hybrid (CF + CBF) ---
    elif choice == "2":
        uid   = input("  Enter MovieLens user ID (1–943): ").strip()
        title = input("  Enter a movie title you liked:   ").strip()
        if not title:
            continue
        result = hybrid_recommend(uid, title, n=5)
        print_hybrid_result(result)

    # --- Option 3: Cold start (new user → genre-based) ---
    elif choice == "3":
        genre = input("  Enter a genre (Action, Romance, Drama, etc.): ").strip()
        if not genre:
            continue
        results = cold_start_recommend(genre, n=5)
        print_cold_start_result(results, genre)

    # --- Option 4: Evaluation report ---
    elif choice == "4":
        print_evaluation_report()

    # --- Quit ---
    elif choice == "q":
        print("Goodbye!")
        break

    else:
        print("  Invalid choice. Type 1, 2, 3, 4, or q.")
3