from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


DATA_DIR = Path("data")
RATINGS_FILE = DATA_DIR / "ratings.csv"
MOVIES_FILE = DATA_DIR / "movies.csv"


def load_data():
    ratings = pd.read_csv(RATINGS_FILE)
    movies = pd.read_csv(MOVIES_FILE)
    return ratings, movies


def filter_data(ratings, min_user_ratings=20, min_movie_ratings=20):
    user_counts = ratings["userId"].value_counts()
    movie_counts = ratings["movieId"].value_counts()

    ratings = ratings[ratings["userId"].isin(user_counts[user_counts >= min_user_ratings].index)]
    ratings = ratings[ratings["movieId"].isin(movie_counts[movie_counts >= min_movie_ratings].index)]

    return ratings.copy()


def train_test_split_ratings(ratings, test_size=0.2, random_state=42):
    train, test = train_test_split(ratings, test_size=test_size, random_state=random_state)
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
    return rmse


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

    if sims.empty:
        return global_mean

    positive = sims[sims > 0]

    if positive.empty:
        return user_ratings.mean()

    aligned_ratings = user_ratings[positive.index]

    numerator = np.sum(positive.values * aligned_ratings.values)
    denominator = np.sum(np.abs(positive.values))

    if denominator == 0:
        return user_ratings.mean()

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


def recommend_movies_for_user(user_id, train, movies, user_movie_matrix, item_similarity, top_n=10):
    if user_id not in user_movie_matrix.index:
        return pd.DataFrame(columns=["movieId", "title", "predicted_rating"])

    rated_movies = set(train[train["userId"] == user_id]["movieId"].unique())
    candidate_movies = [m for m in user_movie_matrix.columns if m not in rated_movies]

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
    rec_df = rec_df.merge(movies[["movieId", "title"]], on="movieId", how="left")
    rec_df = rec_df.sort_values("predicted_rating", ascending=False).head(top_n)

    return rec_df[["movieId", "title", "predicted_rating"]]


def main():
    ratings, movies = load_data()

    print(f"Loaded {len(ratings)} ratings and {len(movies)} movies.")

    ratings = filter_data(ratings)

    print(f"After filtering: {len(ratings)} ratings")
    print(f"Users: {ratings['userId'].nunique()}, Movies: {ratings['movieId'].nunique()}")

    train, test = train_test_split_ratings(ratings)

    print(f"Train size: {len(train)}")
    print(f"Test size: {len(test)}")

    popularity_rmse = evaluate_popularity_baseline(train, test)
    print(f"\nPopularity baseline RMSE: {popularity_rmse:.4f}")

    item_item_rmse, user_movie_matrix, item_similarity = evaluate_item_item_cf(train, test)
    print(f"Item-item collaborative filtering RMSE: {item_item_rmse:.4f}")

    sample_user = train["userId"].iloc[0]
    print(f"\nTop recommendations for user {sample_user}:")

    recs = recommend_movies_for_user(
        sample_user,
        train,
        movies,
        user_movie_matrix,
        item_similarity,
        top_n=10,
    )

    if recs.empty:
        print("No recommendations available for this user.")
    else:
        print(recs.to_string(index=False))


if __name__ == "__main__":
    main()
