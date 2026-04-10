import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Any

logger = logging.getLogger(__name__)

class PMFRecommender:
    """
    Bayesian Probabilistic Matrix Factorization (BPMF) with full Gibbs sampling.
    """
    
    def __init__(
        self,
        n_factors: int = 50,
        n_epochs: int = 120,
        burn_in: int = 30,
        thin: int = 2,
        a0: float = 2.0,
        b0: float = 1.0,
        alpha_u_init: float = 1.0,
        alpha_v_init: float = 1.0,
        alpha_init: float = 1.0,
        validation_split: float = 0.2,
        random_state: int = 42
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.burn_in = burn_in
        self.thin = thin
        self.a0 = a0
        self.b0 = b0
        self.alpha_u = alpha_u_init
        self.alpha_v = alpha_v_init
        self.alpha = alpha_init
        self.val_split = validation_split
        self.random_state = random_state

        self.train_rmse_history: List[float] = []
        self.val_rmse_history: List[float] = []

        self.U_samples: List[np.ndarray] = []
        self.V_samples: List[np.ndarray] = []

        self.U: Optional[np.ndarray] = None
        self.V: Optional[np.ndarray] = None
        
        # 🚀 Running sums to prevent O(N) slowdowns during RMSE calculation
        self._train_pred_sum: Optional[np.ndarray] = None
        self._val_pred_sum: Optional[np.ndarray] = None

    def fit(self, train_matrix_df: pd.DataFrame, val_matrix_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Run the full Gibbs sampler and return the Ensembled Prediction DataFrame.
        """
        try:
            # 1. Save original indices for the final DataFrame output
            self.user_index = train_matrix_df.index
            self.item_columns = train_matrix_df.columns
            
            # 2. Data preparation
            self._prepare_data(train_matrix_df, val_matrix_df)
            rng = np.random.default_rng(self.random_state)

            # 3. Initialization (Pure Random - NO SVD)
            self.U = rng.normal(0, 0.1, (self.n_users, self.n_factors))
            self.V = rng.normal(0, 0.1, (self.n_items, self.n_factors))

            self._build_sparse_indices()

            # 4. Gibbs sampling loop
            logger.info("Starting Gibbs sampling loop...")
            for epoch in range(self.n_epochs):
                self._sample_users(rng)
                self._sample_items(rng)
                self._sample_hyperparameters(rng)

                # Collect samples
                if epoch >= self.burn_in and (epoch - self.burn_in) % self.thin == 0:
                    self.U_samples.append(self.U.copy())
                    self.V_samples.append(self.V.copy())

                # Compute RMSE using running average
                train_rmse, val_rmse = self._compute_rmse(epoch)
                self.train_rmse_history.append(train_rmse)
                self.val_rmse_history.append(val_rmse)

                if epoch % 5 == 0 or epoch == self.n_epochs - 1:
                    phase = "Burn-in" if epoch < self.burn_in else f"Sampling (N={len(self.U_samples)})"
                    logger.info(
                        f"Epoch {epoch+1:3d}/{self.n_epochs} | {phase} | "
                        f"αᵤ={self.alpha_u:.2f} αᵥ={self.alpha_v:.2f} α={self.alpha:.2f} | "
                        f"Train RMSE={train_rmse:.4f} Val RMSE={val_rmse:.4f}"
                    )

            logger.info(f"✅ Training finished. Final validation RMSE = {self.val_rmse_history[-1]:.4f}")
            self._plot_convergence()

            # 5. Build True Bayesian Ensemble Prediction Matrix
            logger.info("🧠 Building final ensemble prediction matrix...")
            if not self.U_samples:
                # Fallback if no samples were collected (epochs < burn_in)
                final_preds = self.U @ self.V.T
            else:
                final_preds = np.zeros((self.n_users, self.n_items))
                for U_s, V_s in zip(self.U_samples, self.V_samples):
                    final_preds += U_s @ V_s.T
                final_preds /= len(self.U_samples)

            # 6. Return exact pipeline-aligned DataFrame (Same as SVD!)
            return pd.DataFrame(final_preds, index=self.user_index, columns=self.item_columns)

        except Exception as e:
            logger.error(f"Fitting failed: {str(e)}", exc_info=True)
            raise

    def _prepare_data(self, train_matrix_df: Any, val_matrix_df: Optional[Any]) -> None:
        R = train_matrix_df.values
        mask = ~np.isnan(R)
        self.R_clean = np.nan_to_num(R, nan=0.0)
        self.n_users, self.n_items = R.shape

        rng = np.random.default_rng(self.random_state)
        self.val_mask = np.zeros_like(mask, dtype=bool)
        for u in range(self.n_users):
            obs = np.where(mask[u])[0]
            if len(obs) == 0:
                continue
            n_val = max(1, int(len(obs) * self.val_split))
            n_val = min(n_val, len(obs))
            val_idx = rng.choice(obs, size=n_val, replace=False)
            self.val_mask[u, val_idx] = True
            
        self.train_mask = mask & (~self.val_mask)

        self.train_users, self.train_items = np.where(self.train_mask)
        self.train_ratings = self.R_clean[self.train_users, self.train_items]
        self.val_users, self.val_items = np.where(self.val_mask)
        self.val_ratings = self.R_clean[self.val_users, self.val_items]

    def _build_sparse_indices(self) -> None:
        self.user_items = [np.where(self.train_mask[u])[0] for u in range(self.n_users)]
        self.item_users = [np.where(self.train_mask[:, i])[0] for i in range(self.n_items)]

    def _sample_users(self, rng: np.random.Generator) -> None:
        for u in range(self.n_users):
            idx = self.user_items[u]
            if len(idx) == 0: continue
            V_u = self.V[idx]
            R_u = self.R_clean[u, idx]
            Lambda = self.alpha_u * np.eye(self.n_factors) + self.alpha * (V_u.T @ V_u)
            try:
                L = np.linalg.cholesky(Lambda)
                y = np.linalg.solve(L, self.alpha * (V_u.T @ R_u))
                mu = np.linalg.solve(L.T, y)
            except np.linalg.LinAlgError:
                jitter = 1e-8 * np.eye(self.n_factors)
                L = np.linalg.cholesky(Lambda + jitter)
                y = np.linalg.solve(L, self.alpha * (V_u.T @ R_u))
                mu = np.linalg.solve(L.T, y)
            z = rng.standard_normal(self.n_factors)
            try:
                v = np.linalg.solve(L.T, z)
                self.U[u] = mu + v
            except np.linalg.LinAlgError:
                self.U[u] = mu

    def _sample_items(self, rng: np.random.Generator) -> None:
        for i in range(self.n_items):
            idx = self.item_users[i]
            if len(idx) == 0: continue
            U_i = self.U[idx]
            R_i = self.R_clean[idx, i]
            Lambda = self.alpha_v * np.eye(self.n_factors) + self.alpha * (U_i.T @ U_i)
            try:
                L = np.linalg.cholesky(Lambda)
                y = np.linalg.solve(L, self.alpha * (U_i.T @ R_i))
                mu = np.linalg.solve(L.T, y)
            except np.linalg.LinAlgError:
                jitter = 1e-8 * np.eye(self.n_factors)
                L = np.linalg.cholesky(Lambda + jitter)
                y = np.linalg.solve(L, self.alpha * (U_i.T @ R_i))
                mu = np.linalg.solve(L.T, y)
            z = rng.standard_normal(self.n_factors)
            try:
                v = np.linalg.solve(L.T, z)
                self.V[i] = mu + v
            except np.linalg.LinAlgError:
                self.V[i] = mu

    def _sample_hyperparameters(self, rng: np.random.Generator) -> None:
        shape_u = self.a0 + (self.n_users * self.n_factors) / 2.0
        rate_u = self.b0 + 0.5 * np.sum(self.U ** 2)
        self.alpha_u = rng.gamma(shape_u, 1.0 / rate_u)

        shape_v = self.a0 + (self.n_items * self.n_factors) / 2.0
        rate_v = self.b0 + 0.5 * np.sum(self.V ** 2)
        self.alpha_v = rng.gamma(shape_v, 1.0 / rate_v)

        resid = 0.0
        for u in range(self.n_users):
            idx = self.user_items[u]
            if len(idx) == 0: continue
            pred = self.U[u] @ self.V[idx].T
            resid += np.sum((self.R_clean[u, idx] - pred) ** 2)
            
        shape_alpha = self.a0 + len(self.train_ratings) / 2.0
        rate_alpha = self.b0 + 0.5 * resid
        self.alpha = rng.gamma(shape_alpha, 1.0 / rate_alpha)

    def _compute_rmse(self, epoch: int) -> Tuple[float, float]:
        """🚀 O(1) Fast Running Sum RMSE Calculation"""
        if epoch < self.burn_in or (epoch - self.burn_in) % self.thin != 0:
            # Burn-in or skipped step: just use current U and V
            train_pred = np.sum(self.U[self.train_users] * self.V[self.train_items], axis=1)
            val_pred = np.sum(self.U[self.val_users] * self.V[self.val_items], axis=1)
        else:
            # Ensemble phase: calculate the prediction for JUST the newest sample
            current_train_pred = np.sum(self.U[self.train_users] * self.V[self.train_items], axis=1)
            current_val_pred = np.sum(self.U[self.val_users] * self.V[self.val_items], axis=1)

            # Add to the running sum
            if self._train_pred_sum is None:
                self._train_pred_sum = current_train_pred.copy()
                self._val_pred_sum = current_val_pred.copy()
            else:
                self._train_pred_sum += current_train_pred
                self._val_pred_sum += current_val_pred

            # Divide by total samples to get the average
            train_pred = self._train_pred_sum / len(self.U_samples)
            val_pred = self._val_pred_sum / len(self.U_samples)

        train_rmse = np.sqrt(np.mean((self.train_ratings - train_pred) ** 2))
        val_rmse = np.sqrt(np.mean((self.val_ratings - val_pred) ** 2))
        return train_rmse, val_rmse

    def _plot_convergence(self) -> None:
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_rmse_history) + 1)
        plt.plot(epochs, self.train_rmse_history, label='Train RMSE')
        plt.plot(epochs, self.val_rmse_history, label='Validation RMSE')
        plt.axvline(x=self.burn_in, color='r', linestyle='--', label='Burn‑in ends')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('BPMF Convergence (Gibbs with hyperpriors)')
        plt.legend()
        plt.grid(True)
        plt.savefig('reports/pmf_convergence.png', dpi=150, bbox_inches='tight')
        plt.close()