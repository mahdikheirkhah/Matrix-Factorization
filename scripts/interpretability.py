import matplotlib.pyplot as plt
import numpy as np
import json
import logging
import pandas as pd
import os
from utils.data_loader import MovieLensLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_5x5_confusion_matrix(actuals, preds, model_name, filename):
    """
    Creates a 5x5 confusion matrix mapping exact star ratings 1-5.
    """
    # Round predictions to the nearest integer and clip to 1-5 range
    y_true = np.array(actuals).astype(int)
    y_pred = np.round(preds).astype(int)
    y_pred = np.clip(y_pred, 1, 5)

    # Calculate the 5x5 matrix
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3, 4, 5])

    # Use a 'YlGnBu' or 'Purples' colormap to see the density clearly
    disp.plot(cmap="YlGnBu", ax=ax, values_format="d")

    plt.title(f"Phase 5 Audit: {model_name} 5x5 Rating Confusion Matrix")
    plt.xlabel("Predicted Star Rating")
    plt.ylabel("Actual Star Rating")

    plt.savefig(f"reports/{filename}")
    plt.close()
    logger.info(f"📊 Saved 5x5 Confusion Matrix: reports/{filename}")


def save_prediction_plot(actuals, preds, model_name, filename):
    """Generic helper to create the Jittered scatter plot."""
    plt.figure(figsize=(10, 6))
    x_jittered = np.array(actuals) + np.random.normal(0, 0.15, len(actuals))

    plt.scatter(
        x_jittered,
        preds,
        alpha=0.5,
        color="#3498db" if model_name == "PMF" else "#e67e22",
        edgecolors="w",
    )
    plt.plot([1, 5], [1, 5], color="red", linestyle="--", label="Ideal")
    plt.xlabel("Actual Ratings (with Jitter)")
    plt.ylabel(f"{model_name} Predicted Ratings")
    plt.title(f"Phase 5: {model_name} Predicted vs. Actual Ratings Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"reports/{filename}")
    plt.close()  # Close to free memory
    logger.info(f"🖼️ Saved: reports/{filename}")


def generate_all_plots(test_df, matrix, user_means, pmf_factors, svd_preds):
    """Generates comparison plots for both PMF and SVD."""
    user_map = {id: i for i, id in enumerate(matrix.index)}
    movie_map = {id: i for i, id in enumerate(matrix.columns.astype(int))}
    U, V = pmf_factors

    pmf_actuals, pmf_preds = [], []
    svd_actuals, svd_preds_list = [], []
    print("🔍 Generating Predicted vs Actual Plots for PMF and SVD...")
    print(
        "⚠️ Note: Only a sample of 10,000 test ratings is plotted for clarity. but metrics are calculated on the full test set. with size:"
        + str(len(test_df))
    )
    sample_df = test_df.sample(min(200000, len(test_df)))

    for _, row in sample_df.iterrows():
        u_id, m_id, actual = int(row["user_id"]), int(row["movie_id"]), row["rating"]
        if u_id in user_map and m_id in movie_map:
            u_idx, m_idx = user_map[u_id], movie_map[m_id]

            # 1. PMF Prediction (Factors + Mean)
            p_pred = np.clip(np.dot(U[u_idx], V[m_idx]) + user_means[u_idx], 1, 5)
            pmf_actuals.append(actual)
            pmf_preds.append(p_pred)

            # 2. SVD Prediction (Residual + Mean)
            s_pred = np.clip(svd_preds[u_idx, m_idx] + user_means[u_idx], 1, 5)
            svd_actuals.append(actual)
            svd_preds_list.append(s_pred)

    save_prediction_plot(pmf_actuals, pmf_preds, "PMF", "predicted_vs_actual_pmf.png")
    save_prediction_plot(
        svd_actuals, svd_preds_list, "SVD", "predicted_vs_actual_svd.png"
    )

    save_5x5_confusion_matrix(pmf_actuals, pmf_preds, "PMF", "confusion_matrix_pmf.png")
    save_5x5_confusion_matrix(
        svd_actuals, svd_preds_list, "SVD", "confusion_matrix_svd.png"
    )


def update_metrics_with_winner(loader):
    """Determines the winner and updates the JSON file."""
    metrics = loader.load_metrics()
    if not metrics:
        return

    svd_rmse = metrics.get("svd", {}).get("rmse", float("inf"))
    pmf_rmse = metrics.get("pmf", {}).get("rmse", float("inf"))

    winner = "PMF" if pmf_rmse < svd_rmse else "SVD"
    improvement = ((svd_rmse - pmf_rmse) / svd_rmse) * 100

    metrics["comparison"] = {
        "winner": winner,
        "improvement_pct": round(improvement, 2),
        "target_met": bool(improvement > 5.0),
    }

    with open("reports/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"🏆 Metric Winner Updated: {winner} ({improvement:.2f}% improvement)")


def analyze_user_recommendations(U, V, matrix, movies_df, test_df, num_users=10):
    """Explains top recommendations for test users."""
    print("\n" + "=" * 75)
    print(f"🔍 LOCAL INTERPRETABILITY: {num_users} RANDOM TEST USERS")
    print("=" * 75)
    test_users = test_df["user_id"].unique()[:num_users]
    user_map = {id: i for i, id in enumerate(matrix.index)}

    for u_id in test_users:
        if u_id not in user_map:
            continue
        u_idx = user_map[u_id]
        user_preds = np.dot(U[u_idx], V.T)
        top_movie_idx = np.argmax(user_preds)
        m_id = int(matrix.columns[top_movie_idx])
        contributions = U[u_idx] * V[top_movie_idx]
        top_factor = np.argmax(contributions)

        title = movies_df[movies_df["movie_id"] == m_id]["title"].values[0]
        print(
            f"👤 User {u_id:4} | 🎬 Rec: {title[:30]:30} | 🔑 Factor {top_factor:2} | ✨ Score: {contributions[top_factor]:.4f}"
        )


def run_interpretability_analysis():
    loader = MovieLensLoader()
    try:
        # Load all artifacts
        U, V = loader.load_pmf_factors()
        svd_preds = np.load("reports/svd_predictions_best.npy")
        matrix = loader.load_user_item_matrix()
        movies_df = loader.load_movies()
        test_df = pd.read_csv("processed/test_ratings.csv")
        user_means = np.load("processed/user_means.npy")

        # 1. Update Metrics
        update_metrics_with_winner(loader)

        # 2. Visualizations for both models
        generate_all_plots(test_df, matrix, user_means, (U, V), svd_preds)

        # 3. Local Report
        analyze_user_recommendations(U, V, matrix, movies_df, test_df)

        logger.info("✅ Phase 5 Interpretability Analysis Complete.")
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")


if __name__ == "__main__":
    run_interpretability_analysis()
