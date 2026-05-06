# ============================================================
# Movie Recommender v3 — Hollywood + Bollywood, Modern Movies
# Datasets: TMDB 5000 (Hollywood) + Bollywood GitHub dataset
# Technique: Content-Based Filtering (TF-IDF + Cosine Similarity)
# ============================================================

import pandas as pd
import json
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# STEP 1: Load and clean the TMDB Hollywood dataset
# ============================================================

print("Loading Hollywood (TMDB) dataset...")

hollywood_raw = pd.read_csv("tmdb_5000_movies.csv")

# The 'genres' column is a JSON string like:
# [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]
# We need to extract just the names → "Action Adventure"

def extract_genre_names(genre_json):
    try:
        genres = ast.literal_eval(genre_json)
        return " ".join([g["name"] for g in genres])
    except:
        return ""

hollywood_raw["genres_clean"] = hollywood_raw["genres"].apply(extract_genre_names)

# Keep only what we need and filter for 2000 onwards
hollywood = pd.DataFrame({
    "title":    hollywood_raw["title"],
    "overview": hollywood_raw["overview"].fillna(""),
    "genres":   hollywood_raw["genres_clean"],
    "year":     pd.to_datetime(
                    hollywood_raw["release_date"], errors="coerce"
                ).dt.year,
    "source":   "Hollywood"
})

hollywood = hollywood[hollywood["year"] >= 2000].dropna(subset=["year"])
print(f"  Hollywood movies (2000+): {len(hollywood)}")


# ============================================================
# STEP 2: Load and clean the Bollywood dataset
# ============================================================

print("Loading Bollywood dataset...")

bollywood_raw = pd.read_csv("bollywood_movies.csv")

# Print columns so we can see what we're working with
print(f"  Bollywood columns: {list(bollywood_raw.columns)}")

# The Bollywood dataset has: Name, Year, Genre, Overview, Director, Cast
# Column names may vary slightly — we handle both cases safely

name_col     = next((c for c in bollywood_raw.columns
                     if c.lower() in ["name", "title", "title_x", "movie_name"]), None)
genre_col    = next((c for c in bollywood_raw.columns
                     if c.lower() in ["genre", "genres"]), None)
overview_col = next((c for c in bollywood_raw.columns
                     if c.lower() in ["overview", "synopsis", "summary", "description"]), None)
year_col     = next((c for c in bollywood_raw.columns
                     if c.lower() in ["year", "year_of_release"]), None)

bollywood = pd.DataFrame({
    "title":    bollywood_raw[name_col] if name_col else "Unknown",
    "overview": bollywood_raw[overview_col].fillna("") if overview_col else "",
    "genres":   bollywood_raw[genre_col].fillna("") if genre_col else "",
    "year":     pd.to_numeric(bollywood_raw[year_col], errors="coerce") if year_col else None,
    "source":   "Bollywood"
})

bollywood = bollywood[bollywood["year"] >= 2000].dropna(subset=["year"])
print(f"  Bollywood movies (2000+): {len(bollywood)}")


# ============================================================
# STEP 3: Merge both datasets into one
# ============================================================

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
# STEP 4: Build the feature string for each movie
# ============================================================
# We combine genres + overview into one text blob per movie.
# More text = better similarity matching.
# Genres are repeated 3x to give them more weight than the overview.

def build_features(row):
    genres   = str(row["genres"]).replace(",", " ").replace("|", " ")
    overview = str(row["overview"])
    # Repeat genres 3 times so genre similarity matters more
    return f"{genres} {genres} {genres} {overview}"

movies["features"] = movies.apply(build_features, axis=1)


# ============================================================
# STEP 5: Build TF-IDF matrix and cosine similarity
# ============================================================

print("Building similarity matrix (this takes a few seconds)...")

tfidf = TfidfVectorizer(
    stop_words="english",   # ignore common words like "the", "a", "is"
    max_features=10000      # limit vocabulary size for speed
)

tfidf_matrix     = tfidf.fit_transform(movies["features"])
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Fast title lookup: lowercase title → dataframe index
title_to_index = pd.Series(
    movies.index,
    index=movies["title"].str.lower().str.strip()
)

print("Ready!\n")


# ============================================================
# STEP 6: The recommendation function
# ============================================================

def find_similar_movies(movie_name: str, n: int = 5,
                        filter_source: str = None):
    """
    Find movies similar to the one the user searched for.

    Parameters:
        movie_name    : str  — what the user typed (partial OK)
        n             : int  — number of results (default 5)
        filter_source : str  — 'Bollywood', 'Hollywood', or None for both

    Returns:
        dict with keys: 'searched_for', 'recommendations'
        OR dict with key: 'error' or 'multiple_matches'
    """

    search = movie_name.strip().lower()

    # --- Match the title ---
    if search in title_to_index:
        idx           = title_to_index[search]
        matched_title = movies.loc[idx, "title"]

    else:
        # Partial match
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

    # --- Apply source filter if requested ---
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
            "source":     row["source"],       # Hollywood or Bollywood
            "similarity": round(score, 2)
        })

    return {
        "searched_for": matched_title,
        "recommendations": results
    }


# ============================================================
# STEP 7: Interactive search loop
# ============================================================

print("=" * 60)
print(" Movie Recommender — Hollywood + Bollywood (2000s+)")
print(" Commands:")
print("   Just type a movie name        → mixed results")
print("   Type a name + ' /b'           → Bollywood only")
print("   Type a name + ' /h'           → Hollywood only")
print("   Type 'quit'                   → exit")
print("=" * 60)

while True:
    user_input = input("\nSearch movie: ").strip()

    if not user_input:
        continue

    if user_input.lower() == "quit":
        print("Goodbye!")
        break

    # Parse filter flag from input
    filter_source = None
    if user_input.endswith(" /b"):
        filter_source = "Bollywood"
        user_input    = user_input[:-3].strip()
    elif user_input.endswith(" /h"):
        filter_source = "Hollywood"
        user_input    = user_input[:-3].strip()

    result = find_similar_movies(user_input, n=5,
                                 filter_source=filter_source)

    # Error
    if "error" in result:
        print(f"\n  {result['error']}")

    # Multiple matches
    elif "multiple_matches" in result:
        print(f"\n  {result['message']}\n")
        for i, title in enumerate(result["multiple_matches"], 1):
            print(f"    {i}. {title}")
        print("\n  Be more specific and search again.")

    # Clean results
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