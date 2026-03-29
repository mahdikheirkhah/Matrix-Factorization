import os
import json
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import mean_squared_error
from models.svd_model import SVDRecommender
from utils.matrix_creation import denormalize_predictions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_svd_pipeline():
    try:
        # 1. Load Data
        train_matrix = pd.read_csv("processed/user_item_matrix.csv", index_col=0)
        test_df = pd.read_csv("processed/test_ratings.csv")

        # 2. Fit Model
        model = SVDRecommender(k=50)
        preds_matrix = model.fit(train_matrix)

        # 3. Export Predictions (Audit Req: reports/svd_predictions.npy)
        os.makedirs("reports", exist_ok=True)
        np.save("reports/svd_predictions.npy", preds_matrix)

        # 4. Calculate RMSE on Test Set
        logger.info("📊 Calculating RMSE on test set...")
        actuals = []
        predictions = []

        # Map IDs to matrix indices
        user_map = {id: i for i, id in enumerate(train_matrix.index)}
        movie_map = {id: i for i, id in enumerate(train_matrix.columns.astype(int))}

        user_means = np.load("processed/user_means.npy") # <--- Add this load

        for _, row in test_df.iterrows():
            u_id, m_id, actual = int(row["user_id"]), int(row["movie_id"]), row["rating"]
            if u_id in user_map and m_id in movie_map:
                u_idx, m_idx = user_map[u_id], movie_map[m_id]
                
                raw_pred = preds_matrix[u_idx, m_idx]
                # Use the helper to get a real star rating
                final_pred = denormalize_predictions(raw_pred, u_idx, user_means)
                
                actuals.append(actual)
                predictions.append(final_pred)

        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        logger.info(f"✨ SVD RMSE: {rmse:.4f}")

        # 5. Append Metrics (Audit Req: reports/model_metrics.json)
        metrics = {"svd": {"rmse": rmse, "k": 50}}
        with open("reports/model_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info("🏁 Phase 3 Exported Successfully.")

    except Exception as e:
        logger.critical(f"❌ SVD Pipeline Failed: {e}")
        raise


if __name__ == "__main__":
    run_svd_pipeline()
