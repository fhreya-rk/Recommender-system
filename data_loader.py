# === FILE: data_loader.py ===
# === PART OF: Movie Recommender System ===
# === IMPORTS FROM: pandas, ast ===

import pandas as pd
import ast


# --- Extract genre names from TMDB's JSON genre column ---
def extract_genre_names(genre_json):
    """Parse a JSON string of genres into a space-separated string."""
    try:
        genres = ast.literal_eval(genre_json)
        return " ".join([g["name"] for g in genres])
    except:
        return ""


# --- Build the combined feature string used for TF-IDF ---
def build_features(row):
    """Combine genres + overview into one text blob per movie.
    Genres are repeated 3x to give them more weight."""
    genres   = str(row["genres"]).replace(",", " ").replace("|", " ")
    overview = str(row["overview"])
    return f"{genres} {genres} {genres} {overview}"


# ============================================================
# Load and clean the TMDB Hollywood dataset
# ============================================================

print("Loading Hollywood (TMDB) dataset...")

hollywood_raw = pd.read_csv("tmdb_5000_movies.csv")

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


# ============================================================
# Load and clean the Bollywood dataset
# ============================================================

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


# ============================================================
# Merge both datasets into one DataFrame
# ============================================================

print("\nMerging datasets...")

movies_df = pd.concat([hollywood, bollywood], ignore_index=True)
movies_df = movies_df.dropna(subset=["title", "genres"])
movies_df = movies_df[movies_df["genres"].str.strip() != ""]
movies_df = movies_df.drop_duplicates(subset=["title"])
movies_df = movies_df.reset_index(drop=True)

# Build the combined feature string for TF-IDF
movies_df["features"] = movies_df.apply(build_features, axis=1)

print(f"Total movies in combined dataset: {len(movies_df)}")
print(f"  Hollywood: {len(movies_df[movies_df['source'] == 'Hollywood'])}")
print(f"  Bollywood: {len(movies_df[movies_df['source'] == 'Bollywood'])}\n")
