import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.data_loader import MovieLensLoader
from utils.matrix_creation import (
    filter_sparse_data,
    create_user_item_matrix,
    normalize_matrix,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_preprocessing():
    try:
        loader = MovieLensLoader()

        # 1. Parse .dat files (using :: separator handled in loader)
        ratings = loader.load_ratings()

        # 2. Clean data: Filter sparse users/movies
        # MovieLens 1M users already have 20+ ratings, but we filter movies for quality
        clean_ratings = filter_sparse_data(
            ratings
        )

        # 3. Split data (Audit Req: random_state=42)
        logger.info("✂️ Splitting data into Train/Test sets...")
        train_df, test_df = train_test_split(
            clean_ratings, test_size=0.15, random_state=42
        )

        # 4. Transform & Handle nulls (Pivot + fillna(0))
        matrix = create_user_item_matrix(train_df)

        # 5. Normalize (Mean Centering)
        norm_matrix, user_means = normalize_matrix(matrix)

        # 6. Save finalized matrix (Audit Req: processed/user_item_matrix.csv)
        os.makedirs("processed", exist_ok=True)
        norm_matrix.to_csv("processed/user_item_matrix.csv")

        # Save test set and means for Phase 4 (Evaluation)
        test_df.to_csv("processed/test_ratings.csv", index=False)
        np.save("processed/user_means.npy", user_means)

        logger.info("🏁 Phase 2 successfully completed!")

    except Exception as e:
        logger.critical(f"❌ Phase 2 failed: {e}")
        raise


if __name__ == "__main__":
    run_preprocessing()
