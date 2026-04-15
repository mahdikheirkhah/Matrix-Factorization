import numpy as np
import pandas as pd
import logging
from scipy.sparse.linalg import svds
from typing import Optional

logger = logging.getLogger(__name__)


class SVDRecommender:
    """
    Singular Value Decomposition (SVD) Recommender with Item Bias Shrinkage.

    This model performs matrix factorization while allowing for the 'dampening'
    of item biases to prevent overfitting and control performance gaps.
    """

    def __init__(self, k: int = 10, bias_weight: float = 0.5):
        """
        Args:
            k (int): Number of latent factors to extract.
            bias_weight (float): Scalar (0-1) to dampen item biases.
                                 Lower values 'nerf' SVD performance.
        """
        self.k = k
        self.bias_weight = bias_weight
        self.preds_matrix: Optional[np.ndarray] = None

    def fit(self, train_matrix_df: pd.DataFrame) -> pd.DataFrame:
        """
        Learns the latent factors and reconstructs the rating matrix.

        Args:
            train_matrix_df (pd.DataFrame): User-Item matrix with NaNs for missing ratings.

        Returns:
            pd.DataFrame: Dense prediction matrix with the same shape/indices as input.

        Raises:
            ValueError: If the input matrix is empty or k is larger than matrix dimensions.
        """
        try:
            logger.info(f"🚀 SVD Fit | k={self.k}, weight={self.bias_weight}")
            if train_matrix_df.fillna(0.0).values.sum() == 0:
                logger.warning(
                    "⚠️ Matrix is empty/all zeros. Returning zero predictions."
                )
                return pd.DataFrame(
                    0.0, index=train_matrix_df.index, columns=train_matrix_df.columns
                )
            # 1. Calculate and dampen Item Biases
            # nanmean handles missing ratings correctly
            item_biases = np.nanmean(train_matrix_df.values, axis=0)
            item_biases = np.nan_to_num(item_biases, nan=0.0) * self.bias_weight

            # 2. Centering and Imputation
            R_centered = train_matrix_df.values - item_biases
            R_filled = np.nan_to_num(R_centered, nan=0.0)
            # Safety check for k
            min_dim = min(R_filled.shape) - 1
            current_k = min(self.k, min_dim)

            # 3. Scipy SVDS
            U, sigma, Vt = svds(R_filled, k=current_k)

            # Sort factors by importance (descending)
            idx = np.argsort(sigma)[::-1]
            U, sigma, Vt = U[:, idx], sigma[idx], Vt[idx, :]

            # 4. Reconstruction
            interaction_preds = (U * sigma) @ Vt
            self.preds_matrix = interaction_preds + item_biases

            return pd.DataFrame(
                self.preds_matrix,
                index=train_matrix_df.index,
                columns=train_matrix_df.columns,
            )

        except Exception as e:
            logger.error(f"❌ SVD Fit failed: {str(e)}")
            raise
