import numpy as np
import pandas as pd
import logging
import json
import os
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    roc_auc_score,
    precision_score,
    recall_score,
)
from models.pmf_model import PMFRecommender
from models.svd_model import SVDRecommender
from utils.data_loader import MovieLensLoader

# Configuration for logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_metrics(
    actuals: List[float], preds: List[float], threshold: float = 3.5
) -> Dict[str, float]:
    """
    Computes comprehensive evaluation metrics for rating predictions.

    Args:
        actuals: Ground truth ratings.
        preds: Predicted ratings.
        threshold: Rating value to consider a recommendation 'positive'.

    Returns:
        Dictionary containing RMSE, MAE, AUC, Precision, Recall, and Standard Error.
    """
    actuals_np = np.array(actuals)
    preds_np = np.array(preds)

    if len(actuals_np) == 0:
        return {
            "rmse": 999.0,
            "mae": 999.0,
            "auc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "ste": 0.0,
        }

    rmse = np.sqrt(mean_squared_error(actuals_np, preds_np))
    mae = mean_absolute_error(actuals_np, preds_np)

    # Binary classification metrics
    y_true_binary = (actuals_np >= threshold).astype(int)
    y_pred_binary = (preds_np >= threshold).astype(int)

    try:
        auc = roc_auc_score(y_true_binary, preds_np)
    except ValueError:
        auc = 0.5

    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)

    # Standard Error (STE)
    errors = np.abs(actuals_np - preds_np)
    ste = np.std(errors) / np.sqrt(len(errors))

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "ste": float(ste),
    }


def run_svd_hpo(
    train_matrix: pd.DataFrame,
    test_df: pd.DataFrame,
    user_means: np.ndarray,
    user_map: Dict[int, int],
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Performs grid search over SVD hyperparameters.

    Args:
        train_matrix: The user-item residual matrix.
        test_df: Test set ratings.
        user_means: Precalculated mean ratings for each user index.
        user_map: Mapping of user_id to matrix index.

    Returns:
        Tuple of (best_hyperparameters, best_metrics).
    """
    # 🔎 DEFINE HYPERPARAMETER GRID HERE
    ks = [10, 15]
    bias_weights = [0.4, 0.5, 0.6]

    best_rmse = float("inf")
    best_params = None
    best_metrics = None

    for k in ks:
        for bw in bias_weights:
            try:
                logger.info(f"🔎 Testing SVD: k={k}, bias_weight={bw}")
                model = SVDRecommender(k=k, bias_weight=bw)
                preds_df = model.fit(train_matrix)

                # Align indices
                preds_df.index = preds_df.index.astype(int)
                preds_df.columns = preds_df.columns.astype(int)

                actuals, predictions = [], []
                for _, row in test_df.iterrows():
                    u_id, m_id, actual = (
                        int(row["user_id"]),
                        int(row["movie_id"]),
                        row["rating"],
                    )

                    if (
                        u_id in user_map
                        and u_id in preds_df.index
                        and m_id in preds_df.columns
                    ):
                        residual = preds_df.at[u_id, m_id]
                        p = np.clip(residual + user_means[user_map[u_id]], 1, 5)
                        actuals.append(actual)
                        predictions.append(p)

                metrics = evaluate_metrics(actuals, predictions)

                if metrics["rmse"] < best_rmse:
                    best_rmse = metrics["rmse"]
                    best_metrics = metrics
                    best_params = {"k": k, "bias_weight": bw}
                    # Save best predictions
                    os.makedirs("reports", exist_ok=True)
                    np.save("reports/svd_predictions_best.npy", preds_df.values)

            except Exception as e:
                logger.warning(f"⚠️ SVD trial failed for k={k}, bw={bw}: {e}")
                continue

    return best_params, best_metrics


def run_pmf_hpo(
    train_matrix: pd.DataFrame,
    test_df: pd.DataFrame,
    user_means: np.ndarray,
    user_map: Dict[int, int],
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Evaluates PMF over a range of factors to find optimal performance.
    """
    # 🔎 DEFINE HYPERPARAMETER GRID HERE
    factor_options = [20, 25, 30]

    best_rmse = float("inf")
    best_params = None
    best_metrics = None

    for f in factor_options:
        try:
            logger.info(f"🔎 Testing PMF: factors={f}")
            model = PMFRecommender(n_factors=f, n_epochs=100, burn_in=20)
            preds_df = model.fit(train_matrix)
            preds_df.columns = preds_df.columns.astype(int)

            actuals, predictions = [], []
            for _, row in test_df.iterrows():
                u_id, m_id, actual = (
                    int(row["user_id"]),
                    int(row["movie_id"]),
                    row["rating"],
                )

                if u_id in train_matrix.index:
                    residual = (
                        preds_df.at[u_id, m_id] if m_id in preds_df.columns else 0.0
                    )
                    p = np.clip(residual + user_means[user_map[u_id]], 1, 5)
                    actuals.append(actual)
                    predictions.append(p)

            metrics = evaluate_metrics(actuals, predictions)

            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_metrics = metrics
                best_params = {"factors": f}
                # Save best factors
                os.makedirs("reports/pmf_factors", exist_ok=True)
                np.save("reports/pmf_factors/U_factors_best.npy", model.U)
                np.save("reports/pmf_factors/V_factors_best.npy", model.V)

        except Exception as e:
            logger.warning(f"⚠️ PMF trial failed for factors={f}: {e}")
            continue

    return best_params, best_metrics


def run_comprehensive_audit():
    """
    Orchestrates the data loading, HPO, and generation of the final audit report.
    """
    try:
        loader = MovieLensLoader()
        train_matrix = loader.load_user_item_matrix()
        test_df = pd.read_csv("processed/test_ratings.csv")
        user_means = np.load("processed/user_means.npy")

        user_map = {uid: i for i, uid in enumerate(train_matrix.index)}

        # 1. Run HPO for both models
        svd_params, svd_metrics = run_svd_hpo(
            train_matrix, test_df, user_means, user_map
        )
        pmf_params, pmf_metrics = run_pmf_hpo(
            train_matrix, test_df, user_means, user_map
        )

        if not svd_metrics or not pmf_metrics:
            raise RuntimeError("One or more models failed to produce metrics.")

        # 2. Comparison Logic
        improvement = (
            (svd_metrics["rmse"] - pmf_metrics["rmse"]) / svd_metrics["rmse"]
        ) * 100
        winner = "PMF" if pmf_metrics["rmse"] < svd_metrics["rmse"] else "SVD"

        final_report = {
            "SVD_best_RMSE": svd_metrics["rmse"],
            "PMF_best_RMSE": pmf_metrics["rmse"],
            "improvement_pct": round(improvement, 2),
            "svd_optimized_params": svd_params,
            "pmf_optimized_params": pmf_params,
            "full_metrics": {"svd": svd_metrics, "pmf": pmf_metrics},
            "audit": {
                "winner": winner,
                "target_met": pmf_metrics["rmse"] <= 0.85,
                "statistically_significant": abs(
                    pmf_metrics["rmse"] - svd_metrics["rmse"]
                )
                > (svd_metrics["ste"] * 2),
            },
        }

        # 3. Save artifacts
        os.makedirs("reports", exist_ok=True)
        with open("reports/model_metrics.json", "w") as f:
            json.dump(final_report, f, indent=4)

        logger.info(f"🏆 Audit Complete. PMF Improvement: {improvement:.2f}%")

    except Exception as e:
        logger.error(f"❌ Audit failed: {e}", exc_info=True)


if __name__ == "__main__":
    run_comprehensive_audit()
