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
            # 1. Edge Case Check: If all values are zero, don't call svds
            if (train_matrix_df.values == 0).all():
                logger.warning(
                    "⚠️ Input matrix is all zeros. Returning zero matrix predictions."
                )
                self.preds_matrix = np.zeros(train_matrix_df.shape)
                return self.preds_matrix

            # 2. Standard SVD Flow
            sparse_matrix = csr_matrix(train_matrix_df.values)
            u, sigma, vt = svds(sparse_matrix, k=self.k)

            sigma_diag = np.diag(sigma)
            self.preds_matrix = np.dot(np.dot(u, sigma_diag), vt)

            logger.info("✅ SVD Decomposition successful.")
            return self.preds_matrix

        except Exception as e:
            logger.error(f"❌ SVD Math Error: {e}")
            raise
