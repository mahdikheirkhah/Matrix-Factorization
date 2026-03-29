import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def filter_sparse_data(df: pd.DataFrame, min_ratings_per_user: int = 5, min_ratings_per_movie: int = 10) -> pd.DataFrame:
    """Filters out sparse users and movies to ensure data quality."""
    logger.info(f"🧹 Filtering sparse data (Min: User={min_ratings_per_user}, Movie={min_ratings_per_movie})")
    try:
        movie_counts = df.groupby("movie_id").size()
        df = df[df["movie_id"].isin(movie_counts[movie_counts >= min_ratings_per_movie].index)]

        user_counts = df.groupby("user_id").size()
        df = df[df["user_id"].isin(user_counts[user_counts >= min_ratings_per_user].index)]

        logger.info(f"✅ Remaining interactions: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"❌ Filtering error: {e}")
        raise

def create_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw ratings into a User-Item Pivot Table, filled with 0s."""
    logger.info("🎬 Creating User-Item matrix...")
    try:
        matrix = df.pivot(index="user_id", columns="movie_id", values="rating").fillna(0)
        logger.info(f"✅ Matrix Shape: {matrix.shape}")
        return matrix
    except Exception as e:
        logger.error(f"❌ Pivot error: {e}")
        raise

def normalize_matrix(matrix: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Subtracts the mean of OBSERVED ratings for each user (ignoring 0s).
    Also logs the sparsity for the Phase 5 Audit report.
    """
    logger.info("⚖️ Starting Matrix Normalization...")
    try:
        matrix_values = matrix.values.copy()
        
        # Calculate Sparsity for Audit Documentation
        total_elements = matrix_values.size
        non_zero_elements = np.count_nonzero(matrix_values)
        sparsity = (1 - (non_zero_elements / total_elements)) * 100
        logger.info(f"📊 Matrix Sparsity: {sparsity:.2f}% ({total_elements - non_zero_elements} zeros)")

        # 1. Mask zeros to calculate true user averages
        temp_matrix = matrix_values.copy().astype(float)
        temp_matrix[temp_matrix == 0] = np.nan
        user_means = np.nanmean(temp_matrix, axis=1)
        
        # Handle edge case: User with no ratings in the training split
        user_means = np.nan_to_num(user_means, nan=0.0)

        mask = matrix_values != 0   # correct mask for all observed entries
        matrix_values[mask] -= user_means[np.where(mask)[0]]

        norm_df = pd.DataFrame(matrix_values, index=matrix.index, columns=matrix.columns)
        logger.info("✅ Normalization complete. Residuals calculated.")
        return norm_df, user_means

    except Exception as e:
        logger.error(f"❌ Normalization error: {e}")
        raise

def denormalize_predictions(pred_value: float, user_idx: int, user_means: np.ndarray) -> float:
    """
    Reverts a centered prediction back to the 1-5 star scale.
    
    Args:
        pred_value: The raw residual predicted by the model.
        user_idx: The index of the user in the matrix.
        user_means: The array of user means saved during preprocessing.
    """
    # 1. Add the user's average rating back to the residual
    full_rating = pred_value + user_means[user_idx]
    
    # 2. Clip to the MovieLens official 1-5 star range
    return np.clip(full_rating, 1, 5)