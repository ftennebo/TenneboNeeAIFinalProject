from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

RATINGS_FILE = DATA_DIR / "ratings.csv"
MOVIES_FILE = DATA_DIR / "movies.csv"


def load_data():
    ratings = pd.read_csv(RATINGS_FILE)
    movies = pd.read_csv(MOVIES_FILE)
    return ratings, movies


def clean_title(title):
    return (
        str(title)
        .lower()
        .replace(",", " ")
        .replace(":", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace("-", " ")
        .strip()
    )


def parse_genres(genre_text):
    if pd.isna(genre_text) or genre_text == "(no genres listed)":
        return set()
    return {genre.strip() for genre in str(genre_text).split("|") if genre.strip()}


def prepare_movies(movies, ratings):
    movies = movies.copy()
    movies["clean_title"] = movies["title"].apply(clean_title)
    movies["genres_text"] = movies["genres"].fillna("").str.replace("|", " ", regex=False)
    movies["content_text"] = movies["clean_title"] + " " + movies["genres_text"]

    rating_summary = ratings.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count"),
    ).reset_index()

    movies = movies.merge(rating_summary, on="movieId", how="left")
    movies["avg_rating"] = movies["avg_rating"].fillna(ratings["rating"].mean())
    movies["rating_count"] = movies["rating_count"].fillna(0)

    return movies


def normalize_series(series):
    series = series.astype(float)
    min_val = series.min()
    max_val = series.max()

    if max_val == min_val:
        return pd.Series(np.ones(len(series)), index=series.index)

    return (series - min_val) / (max_val - min_val)


def build_similarity_data(movies):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(movies["content_text"])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=movies["movieId"],
        columns=movies["movieId"],
    )

    return similarity_df


def find_movie_matches(query, movies, max_results=10):
    query_clean = clean_title(query)
    if not query_clean:
        return pd.DataFrame()

    contains_matches = movies[movies["clean_title"].str.contains(query_clean, na=False)].copy()

    if len(contains_matches) >= max_results:
        return contains_matches.head(max_results)

    movies = movies.copy()
    movies["title_match_score"] = movies["clean_title"].apply(
        lambda x: 1 if query_clean in x else 0
    )

    movies["starts_with_score"] = movies["clean_title"].apply(
        lambda x: 1 if x.startswith(query_clean) else 0
    )

    movies["popularity_score"] = normalize_series(movies["rating_count"])
    movies["rating_score"] = normalize_series(movies["avg_rating"])

    ranked = movies.sort_values(
        ["title_match_score", "starts_with_score", "rating_score", "popularity_score"],
        ascending=[False, False, False, False],
    )

    ranked = ranked[ranked["title_match_score"] > 0]

    if ranked.empty:
        return contains_matches.head(max_results)

    return ranked.head(max_results)


def collect_user_preferences(movies):
    print("\nEnter a few movies you like.")
    print("Type one movie title at a time. Press Enter on a blank line when done.\n")

    selected = []

    while True:
        query = input("Movie title: ").strip()
        if not query:
            break

        matches = find_movie_matches(query, movies)

        if matches.empty:
            print("No matches found. Try another title.\n")
            continue

        print("\nMatches:")
        for i, row in enumerate(matches.itertuples(index=False), start=1):
            print(
                f"{i}. {row.title} [{row.genres}] "
                f"(avg rating: {row.avg_rating:.2f}, ratings: {int(row.rating_count)})"
            )

        choice = input("Choose a number (or press Enter to skip): ").strip()
        if not choice:
            print()
            continue

        if not choice.isdigit():
            print("Invalid choice.\n")
            continue

        choice_idx = int(choice) - 1
        if choice_idx < 0 or choice_idx >= len(matches):
            print("Invalid choice.\n")
            continue

        picked = matches.iloc[choice_idx]
        rating_text = input(f"How much do you like '{picked['title']}'? (0.5 to 5.0): ").strip()

        try:
            rating = float(rating_text)
        except ValueError:
            print("Invalid rating. Using 5.0.\n")
            rating = 5.0

        rating = float(np.clip(rating, 0.5, 5.0))

        selected.append(
            {
                "movieId": picked["movieId"],
                "title": picked["title"],
                "genres": picked["genres"],
                "rating": rating,
            }
        )
        print()

    selected_df = pd.DataFrame(selected)
    if selected_df.empty:
        return selected_df

    return selected_df.drop_duplicates(subset=["movieId"], keep="last")


def build_item_user_rating_data(ratings):
    movie_user_matrix = ratings.pivot_table(
        index="movieId",
        columns="userId",
        values="rating",
    ).fillna(0)

    collaborative_similarity = cosine_similarity(movie_user_matrix)
    collaborative_similarity_df = pd.DataFrame(
        collaborative_similarity,
        index=movie_user_matrix.index,
        columns=movie_user_matrix.index,
    )

    return collaborative_similarity_df


def compute_genre_overlap_score(candidate_genres, selected_movie_rows, selected_ratings):
    candidate_set = parse_genres(candidate_genres)
    if not candidate_set:
        return 0.0

    weighted_overlap = 0.0
    total_weight = 0.0

    for row in selected_movie_rows.itertuples(index=False):
        selected_set = parse_genres(row.genres)
        if not selected_set:
            continue

        overlap = len(candidate_set.intersection(selected_set))
        union = len(candidate_set.union(selected_set))
        similarity = overlap / union if union > 0 else 0.0

        weight = selected_ratings[row.movieId] / 5.0
        weighted_overlap += similarity * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    return weighted_overlap / total_weight


def score_movies(preferences, movies, content_similarity, collaborative_similarity):
    selected_ids = preferences["movieId"].tolist()
    selected_ratings = preferences.set_index("movieId")["rating"]
    selected_movie_rows = movies[movies["movieId"].isin(selected_ids)].copy()

    candidate_movies = movies[~movies["movieId"].isin(selected_ids)].copy()

    genre_scores = []
    content_scores = []
    collaborative_scores = []

    for movie_id in candidate_movies["movieId"]:
        candidate_row = candidate_movies[candidate_movies["movieId"] == movie_id].iloc[0]

        genre_score = compute_genre_overlap_score(
            candidate_row["genres"],
            selected_movie_rows,
            selected_ratings,
        )
        genre_scores.append(genre_score)

        content_total = 0.0
        collaborative_total = 0.0
        weight_total = 0.0

        for selected_id in selected_ids:
            user_rating = selected_ratings[selected_id]
            preference_weight = user_rating / 5.0

            if (
                movie_id in content_similarity.index
                and selected_id in content_similarity.columns
            ):
                content_total += content_similarity.loc[movie_id, selected_id] * preference_weight

            if (
                movie_id in collaborative_similarity.index
                and selected_id in collaborative_similarity.columns
            ):
                collaborative_total += collaborative_similarity.loc[movie_id, selected_id] * preference_weight

            weight_total += preference_weight

        if weight_total == 0:
            content_scores.append(0.0)
            collaborative_scores.append(0.0)
        else:
            content_scores.append(content_total / weight_total)
            collaborative_scores.append(collaborative_total / weight_total)

    candidate_movies["genre_score"] = genre_scores
    candidate_movies["content_score"] = content_scores
    candidate_movies["collaborative_score"] = collaborative_scores
    candidate_movies["popularity_score"] = normalize_series(np.log1p(candidate_movies["rating_count"]))
    candidate_movies["rating_score"] = normalize_series(candidate_movies["avg_rating"])

    candidate_movies["final_score"] = (
        0.40 * candidate_movies["genre_score"]
        + 0.25 * candidate_movies["content_score"]
        + 0.20 * candidate_movies["collaborative_score"]
        + 0.10 * candidate_movies["rating_score"]
        + 0.05 * candidate_movies["popularity_score"]
    )

    candidate_movies = candidate_movies[candidate_movies["genre_score"] > 0]

    return candidate_movies.sort_values(
        ["final_score", "genre_score", "content_score"],
        ascending=False,
    )


def find_hidden_gems(scored_movies):
    gems = scored_movies.copy()

    gems = gems[gems["genre_score"] > 0.15]
    gems = gems[gems["rating_count"] >= 5]
    gems = gems[gems["rating_count"] <= 50]

    gems["low_popularity_score"] = 1 - gems["popularity_score"]

    gems["hidden_gem_score"] = (
        0.45 * gems["genre_score"]
        + 0.25 * gems["content_score"]
        + 0.15 * gems["collaborative_score"]
        + 0.05 * gems["rating_score"]
        + 0.10 * gems["low_popularity_score"]
    )

    gems = gems.sort_values(
        ["hidden_gem_score", "genre_score", "content_score"],
        ascending=False,
    )

    return gems

def main():
    ratings, movies = load_data()
    movies = prepare_movies(movies, ratings)

    print(f"Loaded {len(ratings)} ratings and {len(movies)} movies.")

    content_similarity = build_similarity_data(movies)
    collaborative_similarity = build_item_user_rating_data(ratings)

    preferences = collect_user_preferences(movies)

    if preferences.empty:
        print("\nNo movies were selected, so no recommendations were generated.")
        return

    print("\nYour selected movies:")
    print(preferences[["title", "genres", "rating"]].to_string(index=False))

    scored_movies = score_movies(
        preferences,
        movies,
        content_similarity,
        collaborative_similarity,
    )

    selected_ids = set(preferences["movieId"].tolist())

    scored_movies = scored_movies[~scored_movies["movieId"].isin(selected_ids)].copy()

    recommendations = scored_movies[
        [
            "movieId",
            "title",
            "genres",
            "avg_rating",
            "rating_count",
            "genre_score",
            "content_score",
            "collaborative_score",
            "final_score",
        ]
    ].head(10)

    hidden_gems_source = find_hidden_gems(scored_movies)
    hidden_gems_source = hidden_gems_source[
        ~hidden_gems_source["movieId"].isin(selected_ids)
    ].copy()

    hidden_gems = hidden_gems_source[
        [
            "movieId",
            "title",
            "genres",
            "avg_rating",
            "rating_count",
            "genre_score",
            "content_score",
            "collaborative_score",
            "hidden_gem_score",
        ]
    ].head(5)

    print("\nTop 10 recommendations based on your input:")
    if recommendations.empty:
        print("No recommendations found.")
    else:
        print(recommendations.drop(columns=["movieId"]).to_string(index=False))

    print("\n5 hidden gems matching your tastes:")
    if hidden_gems.empty:
        print("No hidden gems found with the current filters.")
    else:
        print(hidden_gems.drop(columns=["movieId"]).to_string(index=False))
if __name__ == "__main__":
    main()