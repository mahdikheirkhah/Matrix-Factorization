import pandas as pd
import os
import logging

# Configure logger for this module
logger = logging.getLogger(__name__)


class MovieLensLoader:
    """
    Handles loading of the MovieLens 1M dataset with robust error handling.
    """

    def __init__(self, data_path: str = "data/"):
        self.data_path = data_path
        logger.info(f"📁 Initializing MovieLensLoader with data path: {self.data_path}")

    def load_ratings(self) -> pd.DataFrame:
        """
        Loads ratings.dat: UserID::MovieID::Rating::Timestamp

        Returns:
            pd.DataFrame: The ratings dataset.
        """
        path = os.path.join(self.data_path, "ratings.dat")
        logger.info(f"📂 Attempting to load ratings from: {path}")

        try:
            df = pd.read_csv(
                path,
                sep="::",
                engine="python",
                names=["user_id", "movie_id", "rating", "timestamp"],
                encoding="ISO-8859-1",
            )

            if df.empty:
                logger.warning(f"⚠️ The file at {path} is empty.")
            else:
                logger.info(f"✅ Successfully loaded {len(df)} ratings.")

            return df

        except FileNotFoundError:
            logger.error(f"❌ File not found: {path}. Please ensure the data exists.")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"❌ No data found in file: {path}")
            raise
        except Exception as e:
            logger.error(f"❌ An unexpected error occurred while loading ratings: {e}")
            raise

    def load_movies(self) -> pd.DataFrame:
        """
        Loads movies.dat: MovieID::Title::Genres

        Returns:
            pd.DataFrame: The movies dataset.
        """
        path = os.path.join(self.data_path, "movies.dat")
        logger.info(f"📂 Attempting to load movies from: {path}")

        try:
            df = pd.read_csv(
                path,
                sep="::",
                engine="python",
                names=["movie_id", "title", "genres"],
                encoding="ISO-8859-1",
            )

            logger.info(f"✅ Successfully loaded {len(df)} movies.")
            return df

        except FileNotFoundError:
            logger.error(
                f"❌ File not found: {path}. Ensure the MovieLens dataset is extracted."
            )
            raise
        except Exception as e:
            logger.error(f"❌ An unexpected error occurred while loading movies: {e}")
            raise
