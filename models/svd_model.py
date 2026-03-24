import numpy as np
import logging
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

class SVDRecommender:
    def __init__(self, k: int = 50):
        self.k = k
        self.u = None
        self.sigma = None
        self.vt = None
        self.preds_matrix = None

    def fit(self, train_matrix_df):
        """Decomposes the matrix and generates full predictions."""
        logger.info(f"🤖 Starting SVD decomposition with k={self.k}")
        try:
            # Convert to sparse format for scipy compatibility
            sparse_matrix = csr_matrix(train_matrix_df.values)
            
            # SVD Decomposition
            u, sigma, vt = svds(sparse_matrix, k=self.k)
            
            # Convert sigma to diagonal matrix
            sigma_diag = np.diag(sigma)
            
            # Generate Full Predicted Rating Matrix
            # Formula: Predictions = U * Sigma * Vt
            self.preds_matrix = np.dot(np.dot(u, sigma_diag), vt)
            
            logger.info("✅ SVD Decomposition and reconstruction successful.")
            return self.preds_matrix
        except Exception as e:
            logger.error(f"❌ SVD Math Error: {e}")
            raise
