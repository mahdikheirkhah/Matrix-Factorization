import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def generate_recommendations(
    user_id, U, V, user_map, movies_df, matrix, global_mean, movie_means, top_n=10
):
    """
    Generates top-N recommendations for a user by calculating the dot product
    and filtering out already-rated movies.
    """
    # CASE 1: Known User - Standard Matrix Factorization
    if user_id in user_map:
        user_means = np.load("processed/user_means.npy")
        u_idx = user_map[user_id]

        # 1. Get raw residual predictions
        residual_preds = np.dot(U[u_idx], V.T)

        # 2. DENORMALIZE: Add user mean back and clip to 1-5 star range
        preds = np.clip(residual_preds + user_means[u_idx], 1.0, 5.0)

        user_row = matrix.loc[user_id]
        already_rated = user_row[user_row.notna()].index.astype(int).tolist()
        rec_series = pd.Series(preds, index=matrix.columns.astype(int))
        recommendations = rec_series.drop(labels=already_rated, errors="ignore")

    # CASE 2: New User - Fallback to Movie Means (Popularity/Quality Baseline)
    else:
        # Predict using the average of each movie, or global mean if movie is unknown
        recommendations = pd.Series(movie_means, index=matrix.columns.astype(int))
        # Fill any missing movie means with the global average
        recommendations = recommendations.fillna(global_mean)
        # Ensure fallback predictions are also within bounds
        recommendations = np.clip(recommendations, 1.0, 5.0)

    top_ids = recommendations.nlargest(top_n).index
    result = movies_df[movies_df["movie_id"].isin(top_ids)].copy()
    result["predicted_score"] = result["movie_id"].map(recommendations)

    return result.sort_values(by="predicted_score", ascending=False)
