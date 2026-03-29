import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import mean_squared_error
from models.pmf_model import PMFRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pmf_pipeline():
    # 1. Load Data
    train_matrix = pd.read_csv("processed/user_item_matrix.csv", index_col=0)
    test_df = pd.read_csv("processed/test_ratings.csv")
    user_means = np.load("processed/user_means.npy")

    # 2. Train Model
    model = PMFRecommender(n_epochs=100, learning_rate=0.001)  # Use the stable LR
    history = model.fit(train_matrix)

    # 3. Save Convergence Plot
    os.makedirs("reports", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(history, marker="o", color="green")
    plt.title("PMF Convergence (MSE over Epochs)")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.grid(True)
    plt.savefig("reports/pmf_convergence.png")

    # 4. Export Factors
    factor_path = "reports/pmf_factors/"
    os.makedirs(factor_path, exist_ok=True)
    np.save(f"{factor_path}U_factors.npy", model.U)
    np.save(f"{factor_path}V_factors.npy", model.V)

    # 5. Evaluate PMF RMSE (The "Mean Reversion" Step)
    logger.info("📊 Evaluating PMF performance...")
    user_map = {id: i for i, id in enumerate(train_matrix.index)}
    movie_map = {id: i for i, id in enumerate(train_matrix.columns.astype(int))}

    actuals = []
    predictions = []

    # Generate full predictions from factors: R_hat = U * V.T
    full_preds = np.dot(model.U, model.V.T)

    for _, row in test_df.iterrows():
        u_id, m_id, actual = int(row["user_id"]), int(row["movie_id"]), row["rating"]
        if u_id in user_map and m_id in movie_map:
            u_idx = user_map[u_id]
            m_idx = movie_map[m_id]

            # Revert mean and clip to [1, 5]
            pred = np.clip(full_preds[u_idx, m_idx] + user_means[u_idx], 1, 5)
            actuals.append(actual)
            predictions.append(pred)

    pmf_rmse = np.sqrt(mean_squared_error(actuals, predictions))
    logger.info(f"✨ PMF Final RMSE: {pmf_rmse:.4f}")

    # 6. Append to model_metrics.json (Audit Req)
    metrics_file = "reports/model_metrics.json"
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    all_metrics["pmf"] = {
        "rmse": float(pmf_rmse),
        "n_epochs": 50,
        "n_factors": model.n_factors,
    }

    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=4)

    print(f"🏁 Phase 4 Complete. Metrics updated in {metrics_file}")


if __name__ == "__main__":
    run_pmf_pipeline()
