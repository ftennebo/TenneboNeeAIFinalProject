from pathlib import Path

import matplotlib.pyplot as plt
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

    return pd.DataFrame(
        similarity_matrix,
        index=movies["movieId"],
        columns=movies["movieId"],
    )


def build_item_user_rating_data(ratings):
    movie_user_matrix = ratings.pivot_table(
        index="movieId",
        columns="userId",
        values="rating",
    ).fillna(0)

    collaborative_similarity = cosine_similarity(movie_user_matrix)

    return pd.DataFrame(
        collaborative_similarity,
        index=movie_user_matrix.index,
        columns=movie_user_matrix.index,
    )


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


def score_movies(preferences, movies, content_similarity, collaborative_similarity, popularity_weight):
    genre_weight = 0.40
    content_weight = 0.25
    collaborative_weight = 0.20
    rating_weight = 0.10

    # Adjust the other weights proportionally so total stays 1.0
    base_total_without_popularity = genre_weight + content_weight + collaborative_weight + rating_weight
    remaining_weight = 1.0 - popularity_weight
    scale = remaining_weight / base_total_without_popularity

    genre_weight *= scale
    content_weight *= scale
    collaborative_weight *= scale
    rating_weight *= scale

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

            if movie_id in content_similarity.index and selected_id in content_similarity.columns:
                content_total += content_similarity.loc[movie_id, selected_id] * preference_weight

            if movie_id in collaborative_similarity.index and selected_id in collaborative_similarity.columns:
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

    candidate_movies = candidate_movies[candidate_movies["genre_score"] > 0].copy()

    candidate_movies["final_score"] = (
        genre_weight * candidate_movies["genre_score"]
        + content_weight * candidate_movies["content_score"]
        + collaborative_weight * candidate_movies["collaborative_score"]
        + rating_weight * candidate_movies["rating_score"]
        + popularity_weight * candidate_movies["popularity_score"]
    )

    return candidate_movies.sort_values("final_score", ascending=False)


def main():
    ratings, movies = load_data()
    movies = prepare_movies(movies, ratings)

    content_similarity = build_similarity_data(movies)
    collaborative_similarity = build_item_user_rating_data(ratings)

    # Example fixed user taste profile for the experiment
    preferences = pd.DataFrame(
        [
            {"movieId": 260, "title": "Star Wars: Episode IV - A New Hope (1977)", "genres": "Action|Adventure|Sci-Fi", "rating": 5.0},
            {"movieId": 1196, "title": "Star Wars: Episode V - The Empire Strikes Back (1980)", "genres": "Action|Adventure|Sci-Fi", "rating": 5.0},
            {"movieId": 1210, "title": "Star Wars: Episode VI - Return of the Jedi (1983)", "genres": "Action|Adventure|Sci-Fi", "rating": 4.5},
            {"movieId": 1198, "title": "Raiders of the Lost Ark (1981)", "genres": "Action|Adventure", "rating": 4.5},
        ]
    )

    popularity_weights = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
    avg_rating_counts = []

    for pop_weight in popularity_weights:
        scored_movies = score_movies(
            preferences,
            movies,
            content_similarity,
            collaborative_similarity,
            popularity_weight=pop_weight,
        )

        top_10 = scored_movies.head(10)
        avg_rating_counts.append(top_10["rating_count"].mean())

    plt.figure(figsize=(10, 6))
    plt.plot(popularity_weights, avg_rating_counts, marker="o", linewidth=2, color="#4C78A8")
    plt.title("Effect of Popularity Weight on Recommended Movie Popularity", fontsize=16)
    plt.xlabel("Popularity Weight in Final Score", fontsize=12)
    plt.ylabel("Average Number of Ratings in Top 10 Recommendations", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("popularity_weight_experiment.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
