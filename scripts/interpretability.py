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
    """
    Creates and saves a 5x5 confusion matrix for star ratings.
    """
    y_true = np.array(actuals).astype(int)
    y_pred = np.round(np.clip(preds, 1, 5)).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])

    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3, 4, 5])
    disp.plot(cmap="YlGnBu", values_format="d")

    plt.title(f"Rating Consistency: {model_name}")
    plt.tight_layout()
    plt.savefig(f"reports/{filename}.png")
    plt.close()
    logger.info(f"✅ Confusion matrix saved: reports/{filename}.png")


def plot_latent_factors(U: np.ndarray, V: np.ndarray, num_features: int = 10) -> None:
    """
    Visualizes a slice of the learned latent factors to show model interpretability.
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(U[:20, :num_features], cmap="RdBu_r", center=0)
    plt.title("User Latent Features (First 20 Users)")

    plt.subplot(1, 2, 2)
    sns.heatmap(V[:20, :num_features], cmap="RdBu_r", center=0)
    plt.title("Item Latent Features (First 20 Items)")

    plt.tight_layout()
    plt.savefig("reports/latent_factors_heatmap.png")
    plt.close()
    logger.info("✅ Latent factors heatmap saved.")


def generate_audit_summary(metrics: Dict[str, Any]) -> None:
    """
    Generates a clean markdown summary for the project final report.
    """
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
        f"The {'PMF' if pmf_rmse < svd_rmse else 'SVD'} model is the production candidate "
        f"based on lower prediction error and better handling of data sparsity."
    )

    with open("reports/audit_summary.md", "w") as f:
        f.write(summary)
    logger.info("✅ Audit Markdown summary generated.")


def run_interpretability_analysis():
    """
    Main execution flow for generating project reports.
    """
    loader = MovieLensLoader()
    os.makedirs("reports", exist_ok=True)

    try:
        # 1. Load data artifacts
        U, V = loader.load_pmf_factors()
        metrics = loader.load_metrics()  # reports/model_metrics.json
        test_df = pd.read_csv("processed/test_ratings.csv")
        user_means = np.load("processed/user_means.npy")

        # 2. Reconstruct predictions for test set evaluation
        # (Assuming PMF is the primary model for deep analysis)
        user_map = {
            uid: i
            for i, uid in enumerate(
                pd.read_csv("processed/user_item_matrix.csv", index_col=0).index
            )
        }

        actuals, pmf_preds = [], []
        for _, row in test_df.iterrows():
            u_id, m_id, actual = (
                int(row["user_id"]),
                int(row["movie_id"]),
                row["rating"],
            )
            if u_id in user_map:
                u_idx = user_map[u_id]
                # Simplified prediction for interpretability report
                res_pred = np.dot(
                    U[u_idx], V.mean(axis=0)
                )  # Global average fallback for demo
                pmf_preds.append(np.clip(res_pred + user_means[u_idx], 1, 5))
                actuals.append(actual)

        # 3. Generate Visual Reports
        save_comparative_confusion_matrix(
            actuals, pmf_preds, "BPMF Model", "pmf_confusion_matrix"
        )
        plot_latent_factors(U, V)
        generate_audit_summary(metrics)

        logger.info("🚀 All interpretability reports generated successfully.")

    except FileNotFoundError as e:
        logger.error(
            f"❌ Reporting failed: Missing artifact. Did you run hypertunning.py first? {e}"
        )
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    run_interpretability_analysis()
