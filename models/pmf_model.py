import numpy as np
import logging

logger = logging.getLogger(__name__)

class PMFRecommender:
    def __init__(self, n_factors=50, learning_rate=0.01, lambda_reg=0.1, n_epochs=20):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.lambda_reg = lambda_reg
        self.n_epochs = n_epochs
        self.history = []

    def fit(self, train_matrix_df):
        R = train_matrix_df.values
        mask = R > 0
        n_users, n_items = R.shape
        
        # 1. SMALLER INITIALIZATION: 0.1 is often too large; 0.01 is safer
        self.U = np.random.normal(0, 0.01, (n_users, self.n_factors))
        self.V = np.random.normal(0, 0.01, (n_items, self.n_factors))
        
        logger.info(f"🚀 Training PMF: {self.n_epochs} epochs...")
        
        for epoch in range(self.n_epochs):
            # 2. VECTORIZED PREDICTION
            preds = np.dot(self.U, self.V.T)
            
            # 3. CLIP PREDICTIONS: Prevents ratings from hitting +/- Infinity
            preds = np.clip(preds, -10, 10) 
            
            error = (R - preds) * mask
            
            # 4. GRADIENT CALCULATION
            u_grad = -np.dot(error, self.V) + self.lambda_reg * self.U
            v_grad = -np.dot(error.T, self.U) + self.lambda_reg * self.V
            
            # 5. GRADIENT CLIPPING: The "Safety Valve" for memory issues
            np.clip(u_grad, -1, 1, out=u_grad)
            np.clip(v_grad, -1, 1, out=v_grad)
            
            # Update weights
            self.U -= self.lr * u_grad
            self.V -= self.lr * v_grad
            
            # Calculate MSE safely
            mse = np.mean(np.square(error[mask]))
            self.history.append(mse)
            
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch+1} - MSE: {mse:.4f}")
        
        return self.history