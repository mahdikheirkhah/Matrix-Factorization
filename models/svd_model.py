import numpy as np
import pandas as pd
import logging
import os
from scipy.sparse.linalg import svds
from tabulate import tabulate

logger = logging.getLogger(__name__)

class SVDRecommender:
    def __init__(self, k=24):
        self.k = k
        self.preds_matrix = None

    def fit(self, train_matrix_df):
        try:
            logger.info(f"🚀 Running Scipy SVDS | Factors: {self.k}")
            
            # 1. Fill NaNs with 0.0
            R_filled = train_matrix_df.fillna(0.0).values
            
            # 2. Perform exact Scipy SVDS
            U, sigma, Vt = svds(R_filled, k=self.k)
            
            # 3. Reverse for descending order of importance
            U = U[:, ::-1]
            sigma = sigma[::-1]
            Vt = Vt[::-1, :]
            
            # 4. Reconstruct the dense prediction matrix
            sigma_diag = np.diag(sigma)
            self.preds_matrix = np.dot(np.dot(U, sigma_diag), Vt)
            
            # 5. Create DataFrame with proper indices
            preds_df = pd.DataFrame(
                self.preds_matrix, 
                index=train_matrix_df.index, 
                columns=train_matrix_df.columns
            )


            
            logger.info(f"✅ SVD successful. Predictions saved ")
            
            return preds_df

        except Exception as e:
            logger.error(f"❌ SVD Model Error: {str(e)}")
            raise