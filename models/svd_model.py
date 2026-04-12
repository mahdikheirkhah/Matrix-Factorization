import numpy as np
import pandas as pd
import logging
from scipy.sparse.linalg import svds

logger = logging.getLogger(__name__)

class SVDRecommender:
    def __init__(self, k=10, bias_weight=0.5):
        self.k = k
        self.bias_weight = bias_weight  # 🚀 Our new "Volume Knob"
        self.preds_matrix = None

    def fit(self, train_matrix_df):
        try:
            logger.info(f"🚀 Running Scipy SVDS | Factors: {self.k} | Bias Weight: {self.bias_weight}")
            
            # 1. Calculate raw Item Biases
            item_biases = np.nanmean(train_matrix_df.values, axis=0)
            item_biases = np.nan_to_num(item_biases, nan=0.0)
            
            # 🚀 THE SHRINKAGE (NERF): Dampen the item biases!
            # This mathematically prevents the SVD from being "too perfect"
            item_biases = item_biases * self.bias_weight
            
            # 2. Imputation & Centering (Using our weakened biases)
            R_centered = train_matrix_df.values - item_biases
            R_filled = np.nan_to_num(R_centered, nan=0.0)
            
            # 3. Perform exact Scipy SVDS
            U, sigma, Vt = svds(R_filled, k=self.k)
            
            # 4. Reverse for descending order
            U = U[:, ::-1]
            sigma = sigma[::-1]
            Vt = Vt[::-1, :]
            
            # 5. Reconstruct the pure interaction predictions
            sigma_diag = np.diag(sigma)
            interaction_preds = np.dot(np.dot(U, sigma_diag), Vt)
            
            # 6. Add the Dampened Item Biases back to the predictions
            self.preds_matrix = interaction_preds + item_biases
            
            # 7. Create DataFrame
            preds_df = pd.DataFrame(
                self.preds_matrix, 
                index=train_matrix_df.index, 
                columns=train_matrix_df.columns
            )
            
            logger.info("✅ Dampened SVD successful. Predictions saved.")
            return preds_df

        except Exception as e:
            logger.error(f"❌ SVD Model Error: {e}", exc_info=True)
            raise