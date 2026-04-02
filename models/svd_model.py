import numpy as np
import logging

logger = logging.getLogger(__name__)

class SVDRecommender:
    def __init__(self, k=24, lr=0.005, reg=0.02, epochs=50, val_split=0.25, random_state=42):
        self.k = k
        self.lr = lr        
        self.reg = reg      
        self.epochs = epochs
        self.val_split = val_split
        self.random_state = random_state
        self.u = None       
        self.vt = None      
        self.user_biases = None
        self.item_biases = None
        self.global_mean = 0

    def fit(self, train_matrix_df):
        try:
            R = train_matrix_df.values
            n_users, n_items = R.shape
            mask = ~np.isclose(R, 0)
            
            logger.info(f"🚀 Starting Iterative SVD | Users: {n_users} | Items: {n_items} | Factors: {self.k}")

            # 1. Validation Split
            rng = np.random.default_rng(self.random_state)
            val_mask = np.zeros_like(mask, dtype=bool)
            for u in range(n_users):
                obs = np.where(mask[u])[0]
                if len(obs) == 0: continue
                n_val = max(1, int(len(obs) * self.val_split))
                n_val = min(n_val, len(obs))
                val_idx = rng.choice(obs, size=n_val, replace=False)
                val_mask[u, val_idx] = True
                
            train_mask = mask & (~val_mask)
            
            # Extract indices for SGD
            train_users, train_items = np.where(train_mask)
            train_ratings = R[train_users, train_items]
            
            val_users, val_items = np.where(val_mask)
            val_ratings = R[val_users, val_items]

            self.global_mean = np.mean(train_ratings)

            # 2. Initialize factors and biases
            self.u = rng.normal(0, 0.1, (n_users, self.k))
            self.vt = rng.normal(0, 0.1, (self.k, n_items))
            self.user_biases = np.zeros(n_users)
            self.item_biases = np.zeros(n_items)

            for epoch in range(self.epochs):
                total_sq_error = 0
                
                # Shuffle indices for SGD
                indices = np.arange(len(train_ratings))
                rng.shuffle(indices)
                
                for idx in indices:
                    u_idx, i_idx, r = train_users[idx], train_items[idx], train_ratings[idx]
                    
                    pred = (self.global_mean + self.user_biases[u_idx] + 
                            self.item_biases[i_idx] + np.dot(self.u[u_idx], self.vt[:, i_idx]))
                    
                    error = r - pred
                    total_sq_error += error**2
                    
                    # Update parameters
                    self.user_biases[u_idx] += self.lr * (error - self.reg * self.user_biases[u_idx])
                    self.item_biases[i_idx] += self.lr * (error - self.reg * self.item_biases[i_idx])
                    
                    u_old = self.u[u_idx].copy()
                    self.u[u_idx] += self.lr * (error * self.vt[:, i_idx] - self.reg * self.u[u_idx])
                    self.vt[:, i_idx] += self.lr * (error * u_old - self.reg * self.vt[:, i_idx])
                
                # Calculate metrics
                train_rmse = np.sqrt(total_sq_error / len(train_ratings))
                
                # Validation RMSE
                val_preds = (self.global_mean + self.user_biases[val_users] + 
                             self.item_biases[val_items] + 
                             np.sum(self.u[val_users] * self.vt[:, val_items].T, axis=1))
                val_rmse = np.sqrt(np.mean((val_ratings - val_preds)**2))

                if epoch % 5 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs} - Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")

            self.preds_matrix = (self.global_mean + self.user_biases[:, np.newaxis] + 
                                 self.item_biases[np.newaxis, :] + np.dot(self.u, self.vt))
            
            logger.info("✅ SVD Decomposition successful.")
            return self.preds_matrix

        except Exception as e:
            logger.error(f"❌ SVD Model Error: {str(e)}")
            raise