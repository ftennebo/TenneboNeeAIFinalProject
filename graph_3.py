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


def build_content_similarity(movies):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(movies["content_text"])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return pd.DataFrame(
        similarity_matrix,
        index=movies["movieId"],
        columns=movies["movieId"],
    )


def build_collaborative_similarity(ratings):
    movie_user_matrix = ratings.pivot_table(
        index="movieId",
        columns="userId",
        values="rating",
    ).fillna(0)

    similarity_matrix = cosine_similarity(movie_user_matrix)
    return pd.DataFrame(
        similarity_matrix,
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


def score_movies(preferences, movies, content_similarity, collaborative_similarity, method):
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
            preference_weight = selected_ratings[selected_id] / 5.0

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

    if method == "content":
        candidate_movies["final_score"] = (
            0.70 * candidate_movies["genre_score"] +
            0.30 * candidate_movies["content_score"]
        )
    elif method == "collaborative":
        candidate_movies["final_score"] = (
            0.80 * candidate_movies["collaborative_score"] +
            0.10 * candidate_movies["rating_score"] +
            0.10 * candidate_movies["popularity_score"]
        )
    elif method == "hybrid":
        candidate_movies["final_score"] = (
            0.40 * candidate_movies["genre_score"] +
            0.25 * candidate_movies["content_score"] +
            0.20 * candidate_movies["collaborative_score"] +
            0.10 * candidate_movies["rating_score"] +
            0.05 * candidate_movies["popularity_score"]
        )
    else:
        raise ValueError("Unknown method")

    return candidate_movies.sort_values("final_score", ascending=False)


def average_genre_overlap(recommendations):
    return recommendations["genre_score"].mean()


def average_rating_count(recommendations):
    return recommendations["rating_count"].mean()


def main():
    ratings, movies = load_data()
    movies = prepare_movies(movies, ratings)

    content_similarity = build_content_similarity(movies)
    collaborative_similarity = build_collaborative_similarity(ratings)

    preferences = pd.DataFrame(
        [
            {"movieId": 260, "title": "Star Wars: Episode IV - A New Hope (1977)", "genres": "Action|Adventure|Sci-Fi", "rating": 5.0},
            {"movieId": 1196, "title": "Star Wars: Episode V - The Empire Strikes Back (1980)", "genres": "Action|Adventure|Sci-Fi", "rating": 5.0},
            {"movieId": 1, "title": "Toy Story (1995)", "genres": "Adventure|Animation|Children|Comedy|Fantasy", "rating": 4.0},
            {"movieId": 1220, "title": "Blues Brothers, The (1980)", "genres": "Action|Comedy|Musical", "rating": 3.5},
        ]
    )

    methods = ["content", "collaborative", "hybrid"]
    labels = ["Content-Based", "Collaborative", "Hybrid"]

    avg_genre_scores = []
    avg_popularity_scores = []

    for method in methods:
        scored = score_movies(
            preferences,
            movies,
            content_similarity,
            collaborative_similarity,
            method=method,
        ).head(10)

        avg_genre_scores.append(average_genre_overlap(scored))
        avg_popularity_scores.append(average_rating_count(scored))

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(11, 6))

    bars1 = ax1.bar(
        x - width / 2,
        avg_genre_scores,
        width,
        label="Average Genre Overlap",
        color="#4C78A8",
    )
    ax1.set_ylabel("Average Genre Overlap Score")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("Comparison of Recommendation Methods")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        x + width / 2,
        avg_popularity_scores,
        width,
        label="Average Rating Count",
        color="#F28E2B",
    )
    ax2.set_ylabel("Average Number of Ratings")

    for bar in bars1:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    for bar in bars2:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{bar.get_height():.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig("method_comparison_chart.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
