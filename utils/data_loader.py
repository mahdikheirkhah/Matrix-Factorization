import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieLensLoader:
    """
    Handles loading of MovieLens dataset and model artifacts with high modularity.
    """

    def __init__(self, data_path: str = "data/", processed_path: str = "processed/"):
        self.data_path = data_path
        self.processed_path = processed_path
        logger.info(
            f"📁 Loader initialized | Data: {self.data_path} | Processed: {self.processed_path}"
        )

    def _load_dat_file(self, filename: str, columns: list) -> pd.DataFrame:
        """
        Internal global method to load MovieLens .dat files with standard formatting.
        """
        path = os.path.join(self.data_path, filename)
        logger.info(f"📂 Loading: {path}")

        try:
            df = pd.read_csv(
                path,
                sep="::",
                engine="python",
                names=columns,
                encoding="ISO-8859-1",
            )
            if df.empty:
                logger.warning(f"⚠️ {filename} is empty.")
            else:
                logger.info(f"✅ Loaded {len(df)} rows from {filename}")
            return df

        except FileNotFoundError:
            logger.error(f"❌ File not found: {path}")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading {filename}: {str(e)}")
            raise

    def load_ratings(self) -> pd.DataFrame:
        cols = ["user_id", "movie_id", "rating", "timestamp"]
        return self._load_dat_file("ratings.dat", cols)

    def load_movies(self) -> pd.DataFrame:
        cols = ["movie_id", "title", "genres"]
        return self._load_dat_file("movies.dat", cols)

    def load_users(self) -> pd.DataFrame:
        cols = ["user_id", "gender", "age", "occupation", "zip_code"]
        return self._load_dat_file("users.dat", cols)
    
    def load_user_item_matrix(self) -> pd.DataFrame:
        path = os.path.join(self.processed_path, "user_item_matrix.csv")
        try:
            df = pd.read_csv(path, index_col=0)
            
            # 🚀 CRITICAL FIX: Force columns (Movie IDs) and Index (User IDs) to be integers!
            df.columns = df.columns.astype(int)
            df.index = df.index.astype(int)
            
            logger.info(f"✅ Loaded pivot matrix: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"❌ Failed to load matrix at {path}: {e}")
            raise

    def load_pmf_factors(
        self, folder: str = "reports/pmf_factors/"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Loads latent factors U and V as numpy arrays."""
        try:
            u_path = os.path.join(folder, "U_factors_best.npy")
            v_path = os.path.join(folder, "V_factors_best.npy")
            u = np.load(u_path)
            v = np.load(v_path)
            logger.info(f"✅ Loaded Factors | U: {u.shape} | V: {v.shape}")
            return u, v
        except FileNotFoundError as e:
            logger.error(f"❌ Factor files missing in {folder}: {e}")
            raise

    def load_metrics(self, path: str = "reports/model_metrics.json") -> Dict[str, Any]:
        """Loads evaluation metrics from JSON."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
                logger.info(f"✅ Loaded metrics from {path}")
                return data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"❌ Failed to load metrics JSON: {e}")
            return {}
