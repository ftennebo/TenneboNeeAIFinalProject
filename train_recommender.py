import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

RATINGS_FILE = DATA_DIR / "ratings.csv"
MOVIES_FILE = DATA_DIR / "movies.csv"


def load_data():
    ratings = pd.read_csv(RATINGS_FILE)
    movies = pd.read_csv(MOVIES_FILE)
    return ratings, movies


def filter_data(ratings, min_user_ratings=20, min_movie_ratings=20):
    user_counts = ratings["userId"].value_counts()
    movie_counts = ratings["movieId"].value_counts()

    filtered = ratings[
        ratings["userId"].isin(user_counts[user_counts >= min_user_ratings].index)
    ]
    filtered = filtered[
        filtered["movieId"].isin(movie_counts[movie_counts >= min_movie_ratings].index)
    ]

    return filtered.copy()


def train_test_split_ratings(ratings, test_size=0.2, random_state=42):
    train, test = train_test_split(
        ratings,
        test_size=test_size,
        random_state=random_state,
    )
    return train.copy(), test.copy()


def build_user_movie_matrix(ratings):
    return ratings.pivot_table(index="userId", columns="movieId", values="rating")


def evaluate_popularity_baseline(train, test):
    global_mean = train["rating"].mean()
    movie_mean = train.groupby("movieId")["rating"].mean()

    preds = []
    actuals = []

    for row in test.itertuples(index=False):
        pred = movie_mean.get(row.movieId, global_mean)
        preds.append(pred)
        actuals.append(row.rating)

    rmse = np.sqrt(mean_squared_error(actuals, preds))
    return rmse, movie_mean, global_mean


def build_item_similarity_matrix(user_movie_matrix):
    filled = user_movie_matrix.fillna(0)
    similarity = cosine_similarity(filled.T)

    similarity_df = pd.DataFrame(
        similarity,
        index=user_movie_matrix.columns,
        columns=user_movie_matrix.columns,
    )
    return similarity_df


def predict_rating_item_item(user_id, movie_id, user_movie_matrix, item_similarity, global_mean):
    if movie_id not in user_movie_matrix.columns:
        return global_mean

    if user_id not in user_movie_matrix.index:
        return global_mean

    user_ratings = user_movie_matrix.loc[user_id].dropna()
    if user_ratings.empty:
        return global_mean

    if movie_id not in item_similarity.index:
        return global_mean

    sims = item_similarity.loc[movie_id, user_ratings.index]
    positive = sims[sims > 0]

    if positive.empty:
        return float(user_ratings.mean())

    aligned_ratings = user_ratings[positive.index]
    numerator = np.sum(positive.values * aligned_ratings.values)
    denominator = np.sum(np.abs(positive.values))

    if denominator == 0:
        return float(user_ratings.mean())

    pred = numerator / denominator
    return float(np.clip(pred, 0.5, 5.0))


def evaluate_item_item_cf(train, test):
    user_movie_matrix = build_user_movie_matrix(train)
    item_similarity = build_item_similarity_matrix(user_movie_matrix)
    global_mean = train["rating"].mean()

    preds = []
    actuals = []

    for row in test.itertuples(index=False):
        pred = predict_rating_item_item(
            row.userId,
            row.movieId,
            user_movie_matrix,
            item_similarity,
            global_mean,
        )
        preds.append(pred)
        actuals.append(row.rating)

    rmse = np.sqrt(mean_squared_error(actuals, preds))
    return rmse, user_movie_matrix, item_similarity


def get_top_popular_movies(train, movies, top_n=10):
    summary = train.groupby("movieId").agg(
        average_rating=("rating", "mean"),
        rating_count=("rating", "count"),
    )

    summary = summary[summary["rating_count"] >= 30]
    summary = summary.sort_values(
        ["average_rating", "rating_count"],
        ascending=[False, False],
    )

    result = summary.head(top_n).reset_index()
    result = result.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
    return result[["movieId", "title", "genres", "average_rating", "rating_count"]]


def recommend_movies_for_user(
    user_id,
    train,
    movies,
    user_movie_matrix,
    item_similarity,
    top_n=10,
    min_candidate_ratings=30,
):
    if user_id not in user_movie_matrix.index:
        return pd.DataFrame(columns=["movieId", "title", "genres", "predicted_rating"])

    rated_movies = set(train[train["userId"] == user_id]["movieId"].unique())
    movie_counts = train["movieId"].value_counts()

    candidate_movies = [
        movie_id
        for movie_id in user_movie_matrix.columns
        if movie_id not in rated_movies and movie_counts.get(movie_id, 0) >= min_candidate_ratings
    ]

    global_mean = train["rating"].mean()
    recommendations = []

    for movie_id in candidate_movies:
        pred = predict_rating_item_item(
            user_id,
            movie_id,
            user_movie_matrix,
            item_similarity,
            global_mean,
        )
        recommendations.append((movie_id, pred))

    rec_df = pd.DataFrame(recommendations, columns=["movieId", "predicted_rating"])
    rec_df = rec_df.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
    rec_df = rec_df.sort_values("predicted_rating", ascending=False).head(top_n)

    return rec_df[["movieId", "title", "genres", "predicted_rating"]]


def summarize_user_profile(user_id, train, movies, top_n=5):
    user_ratings = train[train["userId"] == user_id].copy()
    if user_ratings.empty:
        return pd.DataFrame(columns=["movieId", "title", "genres", "rating"])

    favorites = user_ratings.sort_values("rating", ascending=False).head(top_n)
    favorites = favorites.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
    return favorites[["movieId", "title", "genres", "rating"]]


def save_results(summary, popular_movies, favorite_movies, recommendations):
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    popular_movies.to_csv(RESULTS_DIR / "top_popular_movies.csv", index=False)
    favorite_movies.to_csv(RESULTS_DIR / "sample_user_favorites.csv", index=False)
    recommendations.to_csv(RESULTS_DIR / "sample_user_recommendations.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a movie recommender.")
    parser.add_argument("--user-id", type=int, default=None, help="User ID for sample recommendations.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of recommendations to display.")
    parser.add_argument("--min-user-ratings", type=int, default=20, help="Minimum ratings per user.")
    parser.add_argument("--min-movie-ratings", type=int, default=20, help="Minimum ratings per movie.")
    args = parser.parse_args()

    ratings, movies = load_data()

    print(f"Loaded {len(ratings)} ratings and {len(movies)} movies.")

    ratings = filter_data(
        ratings,
        min_user_ratings=args.min_user_ratings,
        min_movie_ratings=args.min_movie_ratings,
    )

    print(f"After filtering: {len(ratings)} ratings")
    print(f"Users: {ratings['userId'].nunique()}")
    print(f"Movies: {ratings['movieId'].nunique()}")

    train, test = train_test_split_ratings(ratings)

    print(f"Train size: {len(train)}")
    print(f"Test size: {len(test)}")

    popularity_rmse, _, _ = evaluate_popularity_baseline(train, test)
    print(f"\nPopularity baseline RMSE: {popularity_rmse:.4f}")

    item_item_rmse, user_movie_matrix, item_similarity = evaluate_item_item_cf(train, test)
    print(f"Item-item collaborative filtering RMSE: {item_item_rmse:.4f}")

    sample_user = args.user_id if args.user_id is not None else int(train["userId"].iloc[0])

    print(f"\nSample user: {sample_user}")
    favorite_movies = summarize_user_profile(sample_user, train, movies, top_n=5)
    if favorite_movies.empty:
        print("No favorite-movie summary available for this user.")
    else:
        print("\nSample user's highest-rated movies:")
        print(favorite_movies.to_string(index=False))

    recommendations = recommend_movies_for_user(
        sample_user,
        train,
        movies,
        user_movie_matrix,
        item_similarity,
        top_n=args.top_n,
    )

    print(f"\nTop {args.top_n} recommendations for user {sample_user}:")
    if recommendations.empty:
        print("No recommendations available for this user.")
    else:
        print(recommendations.to_string(index=False))

    popular_movies = get_top_popular_movies(train, movies, top_n=10)

    summary = {
        "dataset": {
            "ratings_count": int(len(ratings)),
            "movies_count": int(ratings["movieId"].nunique()),
            "users_count": int(ratings["userId"].nunique()),
        },
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "evaluation": {
            "popularity_baseline_rmse": float(popularity_rmse),
            "item_item_cf_rmse": float(item_item_rmse),
        },
        "sample_user_id": int(sample_user),
    }

    save_results(summary, popular_movies, favorite_movies, recommendations)
    print(f"\nSaved result files to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
