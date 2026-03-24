import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split

# Configure logger for this module
logger = logging.getLogger(__name__)

def filter_sparse_data(df: pd.DataFrame, min_ratings_per_user: int = 20, min_ratings_per_movie: int = 50) -> pd.DataFrame:
    """
    Filters out users and movies with too few interactions to improve model quality.
    """
    logger.info(f"🧹 Filtering sparse data (Min Ratings: User={min_ratings_per_user}, Movie={min_ratings_per_movie})")
    try:
        # Filter Movies
        movie_counts = df.groupby('movie_id').size()
        df = df[df['movie_id'].isin(movie_counts[movie_counts >= min_ratings_per_movie].index)]
        
        # Filter Users
        user_counts = df.groupby('user_id').size()
        df = df[df['user_id'].isin(user_counts[user_counts >= min_ratings_per_user].index)]
        
        logger.info(f"✅ Filtering complete. Remaining interactions: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"❌ Error during sparse filtering: {e}")
        raise

def create_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw ratings into a User-Item Pivot Table.

    Args:
        df: DataFrame containing ['user_id', 'movie_id', 'rating']
    Returns:
        pd.DataFrame: Pivot table with users as rows and movies as columns.
    """
    logger.info("🎬 Initializing User-Item matrix creation...")

    try:
        if df.empty:
            logger.warning("The input DataFrame is empty. Returning an empty matrix.")
            return pd.DataFrame()

        # Rows = Users, Columns = Movies
        matrix = df.pivot(index="user_id", columns="movie_id", values="rating")

        # Audit requirement: handle nulls (fill with 0 for unrated movies)
        matrix_filled = matrix.fillna(0)

        logger.info(f"✅ Matrix created successfully. Shape: {matrix_filled.shape}")
        return matrix_filled

    except KeyError as e:
        logger.error(f"❌ Column mismatch in input DataFrame: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error during pivot operation: {e}")
        raise


def normalize_matrix(matrix: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Subtracts the mean rating of each user to center the data.

    Args:
        matrix: The filled User-Item DataFrame.
    Returns:
        tuple: (Normalized DataFrame, User Means Array)
    """
    logger.info("⚖️ Starting matrix normalization (Mean Centering)...")

    try:
        if matrix.empty:
            raise ValueError("Cannot normalize an empty matrix.")

        matrix_values = matrix.values

        # Calculate mean for each row (user)
        user_ratings_mean = np.mean(matrix_values, axis=1)

        # Subtract mean (reshaped for broadcasting)
        # We use .reshape(-1, 1) to align the 1D means with the 2D matrix
        matrix_normalized = matrix_values - user_ratings_mean.reshape(-1, 1)

        # Convert back to DataFrame to preserve IDs
        norm_df = pd.DataFrame(
            matrix_normalized, index=matrix.index, columns=matrix.columns
        )

        logger.info("✅ Normalization complete. Data is now mean-centered.")
        return norm_df, user_ratings_mean

    except ValueError as e:
        logger.warning(f"⚠️ Validation error during normalization: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Critical error during normalization math: {e}")
        raise
