import numpy as np
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class PMFRecommender:
    def __init__(
        self,
        n_factors=30,
        learning_rate=0.002,
        lambda_reg=1.0,
        n_epochs=100,
        validation_split=0.1,
        random_state=42,
        early_stopping_patience=30,
        min_delta=1e-4,
        # Adaptive parameters
        momentum_threshold=0.0001,      # min RMSE improvement per epoch
        momentum_boost_factor=2.0,      # multiply LR when stuck
        momentum_decay=0.9,             # momentum coefficient when activated
        max_lr=0.016,                   # maximum learning rate (0.002 -> 0.004 -> 0.008 -> 0.016)
    ):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.initial_lr = learning_rate
        self.lambda_reg = lambda_reg
        self.n_epochs = n_epochs
        self.val_split = validation_split
        self.random_state = random_state
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.momentum_threshold = momentum_threshold
        self.momentum_boost_factor = momentum_boost_factor
        self.momentum_decay = momentum_decay
        self.max_lr = max_lr
        self.history = []
        self.val_history = []

    def fit(self, train_matrix_df, val_matrix_df=None):
        R = train_matrix_df.values
        mask = ~np.isclose(R, 0)
        n_users, n_items = R.shape
        logger.info(f"Starting PMF training | Users: {n_users} | Items: {n_items} | Factors: {self.n_factors}")

        # Validation split
        if val_matrix_df is None:
            rng = np.random.default_rng(self.random_state)
            val_mask = np.zeros_like(mask, dtype=bool)
            for u in range(n_users):
                obs = np.where(mask[u])[0]
                if len(obs) == 0:
                    continue
                n_val = max(1, int(len(obs) * self.val_split))
                n_val = min(n_val, len(obs))
                val_idx = rng.choice(obs, size=n_val, replace=False)
                val_mask[u, val_idx] = True
            train_mask = mask & (~val_mask)
            val_R = R * val_mask
        else:
            val_R = val_matrix_df.values
            train_mask = mask & (val_R == 0)
            val_mask = val_R != 0

        # Random initialisation
        rng = np.random.default_rng(self.random_state)
        scale = 0.5
        self.U = rng.normal(0, scale, (n_users, self.n_factors))
        self.V = rng.normal(0, scale, (n_items, self.n_factors))

        # Momentum state
        U_velocity = np.zeros_like(self.U)
        V_velocity = np.zeros_like(self.V)
        use_momentum = False
        stuck_counter = 0
        prev_val_rmse = None

        best_val_mse = np.inf
        best_epoch = 0
        best_U = None
        best_V = None
        early_stop_counter = 0

        for epoch in range(self.n_epochs):
            # Forward pass
            preds = self.U @ self.V.T
            preds = np.clip(preds, -10, 10)
            train_error = (R - preds) * train_mask

            # Gradients
            u_grad = -train_error @ self.V + self.lambda_reg * self.U
            v_grad = -train_error.T @ self.U + self.lambda_reg * self.V

            # Update step
            if use_momentum:
                U_velocity = self.momentum_decay * U_velocity + self.lr * u_grad
                V_velocity = self.momentum_decay * V_velocity + self.lr * v_grad
                self.U -= U_velocity
                self.V -= V_velocity
            else:
                self.U -= self.lr * u_grad
                self.V -= self.lr * v_grad

            # Compute losses
            train_mse = np.mean(np.square(train_error[train_mask]))
            val_error = (val_R - preds) * val_mask
            val_mse = np.mean(np.square(val_error[val_mask]))
            val_rmse = np.sqrt(val_mse)
            self.history.append(train_mse)
            self.val_history.append(val_mse)

            # --- Adaptive momentum: increase LR and add momentum when stuck ---
            if prev_val_rmse is not None:
                improvement = prev_val_rmse - val_rmse
                # Improvement is ≤ threshold and non‑negative (i.e., no real progress)
                if improvement <= self.momentum_threshold and improvement >= 0:
                    stuck_counter += 1
                    if stuck_counter >= 2:
                        if not use_momentum:
                            use_momentum = True
                            logger.info(f"Stuck (improvement={improvement:.6f}) – enabling momentum")
                        # Increase LR (capped) whenever stuck for 2 consecutive epochs
                        new_lr = min(self.lr * self.momentum_boost_factor, self.max_lr)
                        if new_lr != self.lr:
                            self.lr = new_lr
                            logger.info(f"Stuck – increasing LR to {self.lr:.6f}")
                else:
                    # Significant progress – reset stuck counter but keep momentum and LR
                    stuck_counter = 0
                    if improvement > 0:
                        logger.debug(f"Progress: improvement={improvement:.6f}")
            prev_val_rmse = val_rmse

            # --- Early stopping (based on validation MSE) ---
            if val_mse < best_val_mse - self.min_delta:
                best_val_mse = val_mse
                best_epoch = epoch
                best_U = self.U.copy()
                best_V = self.V.copy()
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch+1} - Train RMSE: {np.sqrt(train_mse):.4f} | Val RMSE: {val_rmse:.4f} | LR: {self.lr:.6f} | Momentum: {use_momentum}")

        # Restore best model
        if best_U is not None:
            self.U = best_U
            self.V = best_V
            logger.info(f"Restored best model from epoch {best_epoch+1} with val RMSE = {np.sqrt(best_val_mse):.4f}")

        self._plot_convergence()
        return self.history

    def _plot_convergence(self):
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.history) + 1)
        plt.plot(epochs, np.sqrt(self.history), label='Train RMSE')
        plt.plot(epochs, np.sqrt(self.val_history), label='Validation RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('PMF Convergence')
        plt.legend()
        plt.grid(True)
        plt.savefig('reports/pmf_convergence.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Saved convergence plot to reports/pmf_convergence.png")