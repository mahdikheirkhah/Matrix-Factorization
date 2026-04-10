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

        # 0. Load Metadata and Ratings
        movies_df = loader.load_movies()
        ratings = loader.load_ratings()
        
        logger.info(f"📊 Initial Movies in metadata: {len(movies_df)}")
        logger.info(f"📊 Initial Ratings count: {len(ratings)}")

        # 1. DATA INTEGRITY CHECK: Ghost Movies
        ghost_ids = ratings[~ratings['movie_id'].isin(movies_df['movie_id'])]['movie_id'].unique()
        if len(ghost_ids) > 0:
            logger.warning(f"👻 Found {len(ghost_ids)} Movie IDs with no metadata! Purging...")
            ratings = ratings[ratings['movie_id'].isin(movies_df['movie_id'])]
        else:
            logger.info("✅ No 'Ghost' Movie IDs found.")

        # 2. DATA INTEGRITY CHECK: Duplicate Titles
        duplicates = movies_df[movies_df.duplicated(subset=['title'], keep=False)]
        if not duplicates.empty:
            logger.info(f"🔍 Found {len(duplicates)} movies with duplicate titles (Inconsistencies):")
            print(duplicates.sort_values('title').head(10))

        # 3. DATA INTEGRITY CHECK: Temporal Consistency (Time Travelers)
        # Extract 4-digit year from title: "Toy Story (1995)" -> 1995
        movies_df['release_year'] = movies_df['title'].str.extract(r'\((\d{4})\)').astype(float)
        
        # Merge temporarily to compare dates
        temp_merged = ratings.merge(movies_df[['movie_id', 'release_year']], on='movie_id')
        temp_merged['rating_year'] = pd.to_datetime(temp_merged['timestamp'], unit='s').dt.year
        
        # Find ratings that happened BEFORE the release year
        time_travelers = temp_merged[temp_merged['rating_year'] < temp_merged['release_year']]
        
        if len(time_travelers) > 0:
            logger.warning(f"⏳ Found {len(time_travelers)} ratings predating movie release! Purging...")
            # Keep only valid temporal ratings
            valid_indices = temp_merged[temp_merged['rating_year'] >= temp_merged['release_year']].index
            ratings = ratings.iloc[valid_indices].reset_index(drop=True)
        else:
            logger.info("✅ All ratings respect temporal logic (no time travelers).")

        # 4. Clean data: Filter sparse users/movies
        clean_ratings = filter_sparse_data(ratings, min_ratings_per_user=100, min_ratings_per_movie=150)

        # 5. Split data (Audit Req: random_state=42)
        logger.info("✂️ Splitting data into Train/Test sets...")
        train_df, test_df = train_test_split(
            clean_ratings, test_size=0.15, random_state=42
        )

        # 6. Transform & Handle nulls (Pivot)
        matrix = create_user_item_matrix(train_df)
        
        # 🚀 Fix the Shape Mismatch Bug
        all_users = np.sort(clean_ratings['user_id'].unique())
        all_movies = np.sort(clean_ratings['movie_id'].unique())
        matrix = matrix.reindex(index=all_users, columns=all_movies)
        
        logger.info(f"✅ Final Matrix Shape: {matrix.shape}")

        # 7. Normalize (Mean Centering)
        norm_matrix, user_means = normalize_matrix(matrix)
        
        # 8. Save finalized artifacts
        os.makedirs("processed", exist_ok=True)
        norm_matrix.to_csv("processed/user_item_matrix.csv")
        test_df.to_csv("processed/test_ratings.csv", index=False)
        np.save("processed/user_means.npy", user_means)

        logger.info("🏁 Phase 2 successfully completed!")

    except Exception as e:
        logger.critical(f"❌ Phase 2 failed: {e}")
        raise


if __name__ == "__main__":
    run_preprocessing()