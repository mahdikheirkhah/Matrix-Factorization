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
        logger.info(f"🤖 Starting SVD decomposition with k={self.k}")
        try:
            if (train_matrix_df.values == 0).all():
                logger.warning("⚠️ Input matrix is all zeros. Returning zero matrix predictions.")
                self.preds_matrix = np.zeros(train_matrix_df.shape)
                return self.preds_matrix

            sparse_matrix = csr_matrix(train_matrix_df.values)
            u, sigma, vt = svds(sparse_matrix, k=self.k)

            # svds returns eigenvalues in ascending order; reverse for descending
            idx = np.argsort(sigma)[::-1]
            self.u = u[:, idx]
            self.sigma = sigma[idx]
            self.vt = vt[idx, :]

            sigma_diag = np.diag(self.sigma)
            self.preds_matrix = np.dot(np.dot(self.u, sigma_diag), self.vt)

            logger.info("✅ SVD Decomposition successful.")
            return self.preds_matrix

        except Exception as e:
            logger.error(f"❌ SVD Math Error: {e}")
            raise