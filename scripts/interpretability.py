import matplotlib.pyplot as plt
import numpy as np
import json
import logging
import pandas as pd
import os
import seaborn as sns
from typing import List, Dict, Any, Tuple
from utils.data_loader import MovieLensLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def save_comparative_confusion_matrix(
    actuals: List[float], preds: List[float], model_name: str, filename: str
) -> None:
    """Creates and saves a 5x5 confusion matrix for star ratings."""
    y_true = np.array(actuals).astype(int)
    y_pred = np.round(np.clip(preds, 1, 5)).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])

    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3, 4, 5])
    disp.plot(cmap="YlGnBu", values_format="d")
    plt.title(f"Rating Consistency: {model_name}")
    plt.tight_layout()
    plt.savefig(f"reports/{filename}.png")
    plt.close()
    logger.info(f"✅ Confusion matrix saved: reports/{filename}.png")


def plot_prediction_drift(
    test_df: pd.DataFrame, pmf_preds: List[float], model_name: str
):
    """
    Creates a chart showing actual ratings vs user IDs.
    Points are colored by the 'drift' (error) from the ideal prediction line.
    """
    plt.figure(figsize=(12, 6))

    # Calculate error (drift)
    actuals = test_df["rating"].values
    drift = np.array(pmf_preds) - actuals

    # Create scatter plot: X = Actual Rating (1-5), Y = User ID
    scatter = plt.scatter(
        actuals,
        test_df["user_id"],
        c=drift,
        cmap="RdBu_r",
        alpha=0.6,
        edgecolors="w",
        label="Model Predictions",
    )

    # Draw the "Ideal Line" (Diagonal representation)
    # Since x is 1-5, we show the vertical segments where predictions should ideally land
    plt.axvline(x=1, color="gray", linestyle="--", alpha=0.3)
    plt.axvline(x=2, color="gray", linestyle="--", alpha=0.3)
    plt.axvline(x=3, color="gray", linestyle="--", alpha=0.3)
    plt.axvline(x=4, color="gray", linestyle="--", alpha=0.3)
    plt.axvline(x=5, color="gray", linestyle="--", alpha=0.3)

    plt.colorbar(scatter, label="Prediction Drift (Positive = Overestimating)")
    plt.title(f"Prediction Accuracy across Users: {model_name}")
    plt.xlabel("Actual Rating (The 'Correct' Line)")
    plt.ylabel("User ID")
    plt.xticks([1, 2, 3, 4, 5])

    plt.tight_layout()
    plt.savefig(f"reports/{model_name.lower()}_drift_chart.png")
    plt.close()
    logger.info(f"✅ Drift chart saved: reports/{model_name.lower()}_drift_chart.png")


def generate_audit_summary(metrics: Dict[str, Any]) -> None:
    """Generates a clean markdown summary for the project final report."""
    svd_rmse = metrics["additional_metrics"]["svd"]["rmse"]
    pmf_rmse = metrics["additional_metrics"]["pmf"]["rmse"]
    improvement = metrics["PMF_vs_SVD_improvement_%"]

    summary = (
        f"# Recommendation System Audit Report\n\n"
        f"## Performance Summary\n"
        f"- **Baseline (SVD) RMSE**: {svd_rmse:.4f}\n"
        f"- **Advanced (PMF) RMSE**: {pmf_rmse:.4f}\n"
        f"- **Relative Improvement**: {improvement}%\n\n"
        f"## Conclusion\n"
        f"The {'PMF' if pmf_rmse < svd_rmse else 'SVD'} model is the production candidate."
    )

    with open("reports/audit_summary.md", "w") as f:
        f.write(summary)
    logger.info("✅ Audit Markdown summary generated.")


def run_interpretability_analysis():
    loader = MovieLensLoader()
    os.makedirs("reports", exist_ok=True)

    try:
        # 1. Load data
        U, V = loader.load_pmf_factors()
        svd_best_preds = np.load("reports/svd_predictions_best.npy")
        metrics = loader.load_metrics()
        test_df = pd.read_csv("processed/test_ratings.csv")
        user_means = np.load("processed/user_means.npy")

        # Load matrix to get mappings
        matrix = loader.load_user_item_matrix()
        user_map = {uid: i for i, uid in enumerate(matrix.index)}
        movie_map = {int(mid): i for i, mid in enumerate(matrix.columns)}

        actuals = []
        pmf_final_preds = []
        svd_final_preds = []

        # 2. Extract and Align Predictions
        for _, row in test_df.iterrows():
            u_id, m_id, actual = (
                int(row["user_id"]),
                int(row["movie_id"]),
                row["rating"],
            )

            if u_id in user_map and m_id in movie_map:
                u_idx, m_idx = user_map[u_id], movie_map[m_id]

                # PMF Prediction (Ensemble mean reconstruction)
                pmf_res = np.dot(U[u_idx], V[m_idx])
                pmf_final_preds.append(np.clip(pmf_res + user_means[u_idx], 1, 5))

                # SVD Prediction (Already denormalized in hypertunning.py)
                svd_val = svd_best_preds[u_idx, m_idx]
                svd_final_preds.append(np.clip(svd_val + user_means[u_idx], 1, 5))

                actuals.append(actual)

        # 3. Generate Reports
        # Confusion Matrices
        save_comparative_confusion_matrix(
            actuals, pmf_final_preds, "PMF Model", "pmf_confusion_matrix"
        )
        save_comparative_confusion_matrix(
            actuals, svd_final_preds, "SVD Model", "svd_confusion_matrix"
        )

        # Drift Charts (The replacement for latent factors)
        plot_prediction_drift(
            test_df[test_df["user_id"].isin(user_map.keys())].head(len(actuals)),
            pmf_final_preds,
            "PMF",
        )
        plot_prediction_drift(
            test_df[test_df["user_id"].isin(user_map.keys())].head(len(actuals)),
            svd_final_preds,
            "SVD",
        )

        generate_audit_summary(metrics)

        logger.info("🚀 All interpretability and comparative reports generated.")

    except Exception as e:
        logger.error(f"❌ Reporting failed: {e}", exc_info=True)


if __name__ == "__main__":
    run_interpretability_analysis()
