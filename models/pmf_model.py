import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Any

logger = logging.getLogger(__name__)

class PMFRecommender:
    """
    Bayesian Probabilistic Matrix Factorization (BPMF) with full Gibbs sampling.
    Implements hierarchical priors on user/item/observation precisions.
    """
    
    def __init__(
        self,
        n_factors: int = 30,
        n_epochs: int = 50,
        burn_in: int = 15,
        thin: int = 1,
        a0: float = 1e-2,
        b0: float = 1e-2,
        alpha_u_init: float = 1.0,
        alpha_v_init: float = 1.0,
        alpha_init: float = 1.0,
        validation_split: float = 0.2,
        random_state: int = 42
    ):
        """
        Parameters
        ----------
        n_factors : int
            Dimensionality of latent factors.
        n_epochs : int
            Number of Gibbs iterations.
        burn_in : int
            Number of iterations to discard before collecting samples.
        thin : int
            Keep every `thin` sample after burn‑in.
        a0, b0 : float
            Shape and rate parameters for Gamma hyperpriors (vague prior).
        alpha_u_init, alpha_v_init, alpha_init : float
            Initial precision values for user features, item features, and observation noise.
        validation_split : float
            Fraction of observed ratings to hold out for internal validation.
        random_state : int
            Seed for reproducibility.
        """
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

        # History for convergence plots
        self.train_rmse_history: List[float] = []
        self.val_rmse_history: List[float] = []

        # Storage for MCMC samples (after burn‑in + thinning)
        self.U_samples: List[np.ndarray] = []   # list of user matrices
        self.V_samples: List[np.ndarray] = []   # list of item matrices

        # Final point estimates (ensemble mean of sampled predictions)
        self.U: Optional[np.ndarray] = None
        self.V: Optional[np.ndarray] = None

        # Internal data placeholders
        self.R_clean: Optional[np.ndarray] = None
        self.train_mask: Optional[np.ndarray] = None
        self.val_mask: Optional[np.ndarray] = None
        self.train_users: Optional[np.ndarray] = None
        self.train_items: Optional[np.ndarray] = None
        self.train_ratings: Optional[np.ndarray] = None
        self.val_users: Optional[np.ndarray] = None
        self.val_items: Optional[np.ndarray] = None
        self.val_ratings: Optional[np.ndarray] = None
        self.user_items: Optional[List[np.ndarray]] = None
        self.item_users: Optional[List[np.ndarray]] = None
        self.n_users: int = 0
        self.n_items: int = 0

    def fit(
        self,
        train_matrix_df: Any,
        val_matrix_df: Optional[Any] = None,
        svd_init: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> List[float]:
        """
        Run the full Gibbs sampler.

        Parameters
        ----------
        train_matrix_df : DataFrame (or any array‑like with .values)
            User‑item matrix with NaNs for missing ratings.
        val_matrix_df : DataFrame, optional
            Explicit validation matrix (if None, internal split is used).
        svd_init : tuple of (U, V) or None
            Warm‑start latent factors from SVD.

        Returns
        -------
        train_rmse_history : List[float]
            RMSE on training set after each epoch (for plotting).
        """
        try:
            # ---------- 1. Data preparation ----------
            self._prepare_data(train_matrix_df, val_matrix_df)
            rng = np.random.default_rng(self.random_state)

            # ---------- 2. Initialization ----------
            self._initialize_latent_factors(svd_init, rng)

            # ---------- 3. Precompute sparse interaction lists ----------
            self._build_sparse_indices()

            # ---------- 4. Gibbs sampling loop ----------
            logger.info("Starting Gibbs sampling loop...")
            for epoch in range(self.n_epochs):
                # Sample user and item latent vectors
                self._sample_users(rng)
                self._sample_items(rng)

                # Sample hyperparameters (precisions)
                self._sample_hyperparameters(rng)

                # Collect samples after burn‑in with thinning
                if epoch >= self.burn_in and (epoch - self.burn_in) % self.thin == 0:
                    self.U_samples.append(self.U.copy())
                    self.V_samples.append(self.V.copy())

                # Compute RMSE using predictive averaging over collected samples
                train_rmse, val_rmse = self._compute_rmse()
                self.train_rmse_history.append(train_rmse)
                self.val_rmse_history.append(val_rmse)

                # Log progress
                if epoch % 5 == 0 or epoch == self.n_epochs - 1:
                    phase = "Burn-in" if epoch < self.burn_in else f"Sampling (samples={len(self.U_samples)})"
                    logger.info(
                        f"Epoch {epoch+1:3d}/{self.n_epochs} | {phase} | "
                        f"αᵤ={self.alpha_u:.3f} αᵥ={self.alpha_v:.3f} α={self.alpha:.3f} | "
                        f"Train RMSE={train_rmse:.4f} Val RMSE={val_rmse:.4f}"
                    )

            # ---------- 5. Final model: ensemble mean of predictions ----------
            # We keep the samples for later prediction. For convenience, we also
            # store the mean of the last sample as a point estimate.
            if self.U_samples:
                self.U = np.mean(self.U_samples, axis=0)
                self.V = np.mean(self.V_samples, axis=0)
            logger.info(f"Training finished. Final validation RMSE = {self.val_rmse_history[-1]:.4f}")
            self._plot_convergence()
            return self.train_rmse_history

        except Exception as e:
            logger.error(f"Fitting failed: {str(e)}", exc_info=True)
            raise

    # ----------------------------------------------------------------------
    # Data preparation methods
    # ----------------------------------------------------------------------
    def _prepare_data(self, train_matrix_df: Any, val_matrix_df: Optional[Any]) -> None:
        """Convert input to numpy, create validation mask, and extract observed ratings."""
        R = train_matrix_df.values
        mask = ~np.isnan(R)
        self.R_clean = np.nan_to_num(R, nan=0.0)
        self.n_users, self.n_items = R.shape

        if val_matrix_df is not None:
            # Use externally provided validation set
            valR = val_matrix_df.values
            self.val_mask = ~np.isnan(valR) & mask   # only where both have observations
            self.train_mask = mask & (~self.val_mask)
            logger.info("Using external validation matrix.")
        else:
            # Internal random split
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

        # Flatten indices for fast RMSE calculation
        self.train_users, self.train_items = np.where(self.train_mask)
        self.train_ratings = self.R_clean[self.train_users, self.train_items]
        self.val_users, self.val_items = np.where(self.val_mask)
        self.val_ratings = self.R_clean[self.val_users, self.val_items]

        logger.info(f"Data: {self.n_users} users, {self.n_items} items, "
                    f"train obs={len(self.train_ratings)}, val obs={len(self.val_ratings)}")

    def _build_sparse_indices(self) -> None:
        """Precompute lists of rated items per user and rated users per item."""
        self.user_items = [np.where(self.train_mask[u])[0] for u in range(self.n_users)]
        self.item_users = [np.where(self.train_mask[:, i])[0] for i in range(self.n_items)]
        logger.info("Sparse interaction indices built.")

    # ----------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------
    def _initialize_latent_factors(
        self, svd_init: Optional[Tuple[np.ndarray, np.ndarray]], rng: np.random.Generator
    ) -> None:
        """Initialize U and V either from SVD warm‑start or random Gaussian."""
        if svd_init is not None:
            self.U = svd_init[0].copy()
            self.V = svd_init[1].copy()
            # Ensure factor dimension matches
            if self.U.shape[1] != self.n_factors:
                logger.warning(f"SVD init dimension {self.U.shape[1]} != {self.n_factors}, truncating/padding.")
                # Simple truncation or zero padding (not rigorous, but works for demo)
                if self.U.shape[1] > self.n_factors:
                    self.U = self.U[:, :self.n_factors]
                    self.V = self.V[:, :self.n_factors]
                else:
                    pad_u = np.zeros((self.U.shape[0], self.n_factors - self.U.shape[1]))
                    pad_v = np.zeros((self.V.shape[0], self.n_factors - self.V.shape[1]))
                    self.U = np.hstack([self.U, pad_u])
                    self.V = np.hstack([self.V, pad_v])
            logger.info("Initialized with SVD warm‑start.")
        else:
            self.U = rng.normal(0, 0.1, (self.n_users, self.n_factors))
            self.V = rng.normal(0, 0.1, (self.n_items, self.n_factors))
            logger.info("Initialized with random Gaussian.")

    # ----------------------------------------------------------------------
    # Gibbs sampling core
    # ----------------------------------------------------------------------
    def _sample_users(self, rng: np.random.Generator) -> None:
        """
        Sample each user vector from its multivariate normal conditional posterior.
        Uses Cholesky for numerical stability.
        """
        for u in range(self.n_users):
            idx = self.user_items[u]
            if len(idx) == 0:
                continue

            V_u = self.V[idx]                     # (n_rated, D)
            R_u = self.R_clean[u, idx]            # (n_rated,)

            # Posterior precision matrix
            Lambda = self.alpha_u * np.eye(self.n_factors) + self.alpha * (V_u.T @ V_u)

            # Compute mean = alpha * Sigma * (V_u^T R_u) using Cholesky
            try:
                L = np.linalg.cholesky(Lambda)          # lower triangular
                # Solve L * y = (alpha * V_u^T R_u)  -> y
                y = np.linalg.solve(L, self.alpha * (V_u.T @ R_u))
                # Solve L^T * mu = y  -> mu
                mu = np.linalg.solve(L.T, y)
            except np.linalg.LinAlgError:
                # Fallback: add small jitter and retry
                jitter = 1e-8 * np.eye(self.n_factors)
                Lambda_jitter = Lambda + jitter
                L = np.linalg.cholesky(Lambda_jitter)
                y = np.linalg.solve(L, self.alpha * (V_u.T @ R_u))
                mu = np.linalg.solve(L.T, y)

            # Sample from N(mu, Sigma) where Sigma = Lambda^{-1}
            # Compute Sigma via Cholesky of Lambda, then sample
            # We already have L such that L L^T = Lambda. Then Sigma = (L L^T)^{-1} = L^{-T} L^{-1}
            # To sample: z ~ N(0,I), then mu + L^{-T} z
            z = rng.standard_normal(self.n_factors)
            try:
                # Solve L^T v = z  -> v = L^{-T} z
                v = np.linalg.solve(L.T, z)
                self.U[u] = mu + v
            except np.linalg.LinAlgError:
                # If singular, just use mean (should not happen with jitter)
                self.U[u] = mu

    def _sample_items(self, rng: np.random.Generator) -> None:
        """Symmetric to _sample_users."""
        for i in range(self.n_items):
            idx = self.item_users[i]
            if len(idx) == 0:
                continue

            U_i = self.U[idx]
            R_i = self.R_clean[idx, i]

            Lambda = self.alpha_v * np.eye(self.n_factors) + self.alpha * (U_i.T @ U_i)

            try:
                L = np.linalg.cholesky(Lambda)
                y = np.linalg.solve(L, self.alpha * (U_i.T @ R_i))
                mu = np.linalg.solve(L.T, y)
            except np.linalg.LinAlgError:
                jitter = 1e-8 * np.eye(self.n_factors)
                Lambda_jitter = Lambda + jitter
                L = np.linalg.cholesky(Lambda_jitter)
                y = np.linalg.solve(L, self.alpha * (U_i.T @ R_i))
                mu = np.linalg.solve(L.T, y)

            z = rng.standard_normal(self.n_factors)
            try:
                v = np.linalg.solve(L.T, z)
                self.V[i] = mu + v
            except np.linalg.LinAlgError:
                self.V[i] = mu

    def _sample_hyperparameters(self, rng: np.random.Generator) -> None:
        """
        Sample αᵤ, αᵥ, α from their Gamma conditional posteriors.
        """
        # --- Sample αᵤ (user precision) ---
        shape_u = self.a0 + (self.n_users * self.n_factors) / 2.0
        rate_u = self.b0 + 0.5 * np.sum(self.U ** 2)
        self.alpha_u = rng.gamma(shape_u, 1.0 / rate_u)   # gamma(shape, scale)

        # --- Sample αᵥ (item precision) ---
        shape_v = self.a0 + (self.n_items * self.n_factors) / 2.0
        rate_v = self.b0 + 0.5 * np.sum(self.V ** 2)
        self.alpha_v = rng.gamma(shape_v, 1.0 / rate_v)

        # --- Sample α (observation precision) ---
        # Compute sum of squared residuals over training set
        resid = 0.0
        for u in range(self.n_users):
            idx = self.user_items[u]
            if len(idx) == 0:
                continue
            pred = self.U[u] @ self.V[idx].T
            resid += np.sum((self.R_clean[u, idx] - pred) ** 2)
        shape_alpha = self.a0 + len(self.train_ratings) / 2.0
        rate_alpha = self.b0 + 0.5 * resid
        self.alpha = rng.gamma(shape_alpha, 1.0 / rate_alpha)

    # ----------------------------------------------------------------------
    # Prediction and evaluation
    # ----------------------------------------------------------------------
    def _compute_rmse(self) -> Tuple[float, float]:
        """
        Compute train and validation RMSE using predictive averaging
        over all collected MCMC samples (after burn‑in & thinning).
        """
        if not self.U_samples:
            # No samples yet: use current point estimates
            train_pred = np.sum(self.U[self.train_users] * self.V[self.train_items], axis=1)
            val_pred = np.sum(self.U[self.val_users] * self.V[self.val_items], axis=1)
        else:
            # Average predictions over all stored samples
            train_pred_sum = np.zeros(len(self.train_ratings))
            val_pred_sum = np.zeros(len(self.val_ratings))
            for U_s, V_s in zip(self.U_samples, self.V_samples):
                train_pred_sum += np.sum(U_s[self.train_users] * V_s[self.train_items], axis=1)
                val_pred_sum += np.sum(U_s[self.val_users] * V_s[self.val_items], axis=1)
            train_pred = train_pred_sum / len(self.U_samples)
            val_pred = val_pred_sum / len(self.U_samples)

        train_rmse = np.sqrt(np.mean((self.train_ratings - train_pred) ** 2))
        val_rmse = np.sqrt(np.mean((self.val_ratings - val_pred) ** 2))
        return train_rmse, val_rmse

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """
        Predict ratings for given (user, item) pairs using the ensemble mean.

        Parameters
        ----------
        user_ids, item_ids : array-like of ints
            Indices of users and items.

        Returns
        -------
        predictions : np.ndarray
            Predicted rating for each pair.
        """
        if not self.U_samples:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        # Average predictions over all stored samples
        pred_sum = np.zeros(len(user_ids))
        for U_s, V_s in zip(self.U_samples, self.V_samples):
            pred_sum += np.sum(U_s[user_ids] * V_s[item_ids], axis=1)
        return pred_sum / len(self.U_samples)

    # ----------------------------------------------------------------------
    # Plotting
    # ----------------------------------------------------------------------
    def _plot_convergence(self) -> None:
        """Plot training and validation RMSE over epochs."""
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
        plt.savefig('reports/bpmf_convergence.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Convergence plot saved to reports/bpmf_convergence.png")