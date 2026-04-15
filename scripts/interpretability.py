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


def save_rmse_comparison(metrics: Dict[str, Any]):
    """Issue #2: Generate a bar chart comparing SVD and PMF RMSE."""
    models = ["SVD (Baseline)", "PMF (Advanced)"]
    rmses = [
        metrics["additional_metrics"]["svd"]["rmse"],
        metrics["additional_metrics"]["pmf"]["rmse"],
    ]

    plt.figure(figsize=(8, 6))
    colors = ["#95a5a6", "#2ecc71"]  # Gray for baseline, Green for winner
    bars = plt.bar(models, rmses, color=colors, width=0.6)

    plt.ylabel("RMSE (Lower is Better)")
    plt.title("Final Model Performance Comparison")
    plt.ylim(0, max(rmses) * 1.2)

    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.01,
            round(yval, 4),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.savefig("reports/rmse_comparison.png")
    plt.close()
    logger.info("✅ RMSE comparison bar chart saved.")


def explain_local_prediction(
    U: np.ndarray,
    V: np.ndarray,
    user_idx: int,
    movie_idx: int,
    user_id: int,
    movies_df: pd.DataFrame,
):
    """Issue #3: Explain why a specific prediction happened (Local Interpretability)."""
    # Dot product components
    contributions = U[user_idx] * V[movie_idx]
    top_factor = np.argmax(np.abs(contributions))

    movie_title = movies_df.iloc[movie_idx]["title"]

    explanation = (
        f"--- Local Interpretability ---\n"
        f"User {user_id} -> Movie: {movie_title}\n"
        f"Strongest Driver: Latent Factor {top_factor}\n"
        f"Factor Contribution: {contributions[top_factor]:.4f}\n"
    )
    return explanation


def analyze_global_trends(
    V: np.ndarray, movies_df: pd.DataFrame, top_k_factors: int = 3
):
    """Issue #4: Find movies that define the top latent factors (Global Trends)."""
    trends_report = "--- Global Latent Factor Trends ---\n"

    for f in range(top_k_factors):
        # Find indices of movies with highest weight in this factor
        top_movie_indices = np.argsort(V[:, f])[-5:][::-1]
        titles = movies_df.iloc[top_movie_indices]["title"].values

        trends_report += f"\nFactor {f} (Representative Movies):\n"
        for i, title in enumerate(titles):
            trends_report += f"  {i+1}. {title}\n"

    return trends_report


def save_comparative_confusion_matrix(
    actuals: List[float], preds: List[float], model_name: str, filename: str
) -> None:
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


def plot_predicted_vs_actual(actuals, preds):
    plt.figure(figsize=(8, 6))
    plt.scatter(actuals, preds, alpha=0.5, color="blue")
    plt.plot([1, 5], [1, 5], "r--")  # The perfect prediction line
    plt.xlabel("Actual Ratings")
    plt.ylabel("Predicted Ratings")
    plt.title("Predicted vs Actual Ratings (PMF)")
    plt.savefig("reports/predicted_vs_actual.png")
    plt.close()


def run_interpretability_analysis():
    loader = MovieLensLoader()
    os.makedirs("reports", exist_ok=True)

    try:
        # 1. Load Data
        U, V = loader.load_pmf_factors()
        metrics = loader.load_metrics()
        movies_df = loader.load_movies()
        matrix = loader.load_user_item_matrix()
        user_means = np.load("processed/user_means.npy")
        test_df = pd.read_csv("processed/test_ratings.csv")

        user_map = {uid: i for i, uid in enumerate(matrix.index)}
        movie_map = {int(mid): i for i, mid in enumerate(matrix.columns)}

        # 2. Performance Reporting (Issue #2)
        save_rmse_comparison(metrics)

        # 3. Local/Global Text Audit (Issues #3 & #4)
        # Pick a sample from test set for local explanation
        sample_row = test_df.iloc[0]
        u_id, m_id = int(sample_row["user_id"]), int(sample_row["movie_id"])
        u_idx, m_idx = user_map[u_id], movie_map[m_id]

        local_info = explain_local_prediction(U, V, u_idx, m_idx, u_id, movies_df)
        global_info = analyze_global_trends(V, movies_df)

        # Append to audit summary
        with open("reports/audit_summary.md", "a") as f:
            f.write(f"\n## Interpretability Analysis\n")
            f.write(f"### Global Trends (Factor Analysis)\n```\n{global_info}```\n")
            f.write(f"### Local Example\n```\n{local_info}```\n")

        # 4. Generate Confusion Matrices
        # (Assuming PMF predictions alignment for brevity)
        actuals, pmf_final_preds = [], []
        for _, row in test_df.head(500).iterrows():  # Sample for speed
            uid, mid, act = int(row["user_id"]), int(row["movie_id"]), row["rating"]
            if uid in user_map and mid in movie_map:
                p = (
                    np.dot(U[user_map[uid]], V[movie_map[mid]])
                    + user_means[user_map[uid]]
                )
                pmf_final_preds.append(np.clip(p, 1, 5))
                actuals.append(act)

        save_comparative_confusion_matrix(
            actuals, pmf_final_preds, "PMF Model", "pmf_confusion_matrix"
        )
        plot_predicted_vs_actual(actuals, pmf_final_preds)

        logger.info(
            "🚀 All 5 Audit Issues have been addressed and documented in /reports."
        )

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}", exc_info=True)


if __name__ == "__main__":
    run_interpretability_analysis()
