import os
import json
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, precision_score, recall_score
from models.pmf_model import PMFRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_metrics(actuals, preds, threshold=3.5):
    """Calculates comprehensive metrics for the audit."""
    actuals = np.array(actuals)
    preds = np.array(preds)
    
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae = mean_absolute_error(actuals, preds)
    
    y_true_binary = (actuals >= threshold).astype(int)
    y_pred_binary = (preds >= threshold).astype(int)
    
    auc = roc_auc_score(y_true_binary, preds)
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)

    errors = np.abs(actuals - preds)
    ste = np.std(errors) / np.sqrt(len(errors))
    
    return {
        "rmse": float(rmse), "mae": float(mae), "auc": float(auc),
        "precision": float(precision), "recall": float(recall), "ste": float(ste)
    }

def run_pmf_pipeline():
    # 1. Load Data
    logger.info("📂 Loading Data...")
    train_matrix = pd.read_csv("processed/user_item_matrix.csv", index_col=0)
    test_df = pd.read_csv("processed/test_ratings.csv")
    user_means = np.load("processed/user_means.npy")

    # ---------------------------------------------------------
    # NEW: Data Distribution Analysis
    # ---------------------------------------------------------
    logger.info("📈 Analyzing Train Matrix Distribution...")
    
    # Extract only valid (non-NaN) values from the matrix
    valid_ratings = train_matrix.values[~np.isnan(train_matrix.values)]
    
    # Calculate basic statistics
    mean_val = np.mean(valid_ratings)
    std_val = np.std(valid_ratings)
    skewness = pd.Series(valid_ratings).skew() # Skewness around 0 means Normal
    
    logger.info(f"📊 Stats | Mean: {mean_val:.4f} | Std: {std_val:.4f} | Skewness: {skewness:.4f}")
    
    # Create the Distribution Plot
    plt.figure(figsize=(10, 6))
    sns.histplot(valid_ratings, bins=50, kde=True, color='royalblue', stat='density')
    
    # Add mean and standard deviation lines for visual reference
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean ({mean_val:.2f})')
    plt.axvline(mean_val + std_val, color='green', linestyle='dotted', linewidth=2, label='+1 Std Dev')
    plt.axvline(mean_val - std_val, color='green', linestyle='dotted', linewidth=2, label='-1 Std Dev')
    
    plt.title("Distribution of Mean-Centered Ratings in Train Matrix", fontsize=14)
    plt.xlabel("Centered Rating Value (Actual Rating - User Mean)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    os.makedirs("reports", exist_ok=True)
    plot_path = "reports/train_data_distribution.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Saved distribution plot to {plot_path}")
    # ---------------------------------------------------------

    # 2. Load SVD Warm-Start Factors (Currently Disabled in pipeline, but loaded just in case)
    try:
        U_svd = np.load("reports/svd_U_init.npy")
        V_svd = np.load("reports/svd_V_init.npy")
        svd_init = (U_svd, V_svd)
        logger.info(f"🔥 Loaded SVD factors for warm-start (shape: U={U_svd.shape}, V={V_svd.shape})")
    except FileNotFoundError:
        svd_init = None
        logger.warning("⚠️ No SVD factors found. Using random initialization.")

    # 3. Initialize and Train PMF Model
    # 3. Initialize and Train PMF Model
    logger.info("🚀 Initializing PMF Recommender...")
    model = PMFRecommender(
    n_factors=20,
    n_epochs=600,
    burn_in=50,
    thin=2,
    a0=0.5, b0=0.5,
    alpha_u_init=1.0, alpha_v_init=1.0, alpha_init=1.0,
    validation_split=0.2
    )
    # Train from scratch (svd_init=None per your recent architecture)
    _ = model.fit(train_matrix, svd_init=None)

    # 4. Export the Best PMF Factors
    factor_path = "reports/pmf_factors/"
    os.makedirs(factor_path, exist_ok=True)
    np.save(f"{factor_path}U_factors_best.npy", model.U)
    np.save(f"{factor_path}V_factors_best.npy", model.V)

    # 5. Evaluate PMF Performance on Test Set
    logger.info("📊 Evaluating PMF performance on Test Set...")
    user_map = {id: i for i, id in enumerate(train_matrix.index)}
    movie_map = {id: i for i, id in enumerate(train_matrix.columns.astype(int))}

    actuals = []
    predictions = []
    full_preds = np.dot(model.U, model.V.T)

    for _, row in test_df.iterrows():
        u_id, m_id, actual = int(row["user_id"]), int(row["movie_id"]), row["rating"]
        if u_id in user_map and m_id in movie_map:
            u_idx = user_map[u_id]
            m_idx = movie_map[m_id]

            # Revert mean centering and clip to [1, 5]
            pred = np.clip(full_preds[u_idx, m_idx] + user_means[u_idx], 1, 5)
            actuals.append(actual)
            predictions.append(pred)

    pmf_metrics = evaluate_metrics(actuals, predictions)
    pmf_rmse = pmf_metrics['rmse']
    logger.info(f"✨ PMF Final Test RMSE: {pmf_rmse:.4f}")

    # 6. Update model_metrics.json (Audit Requirement)
    metrics_file = "reports/model_metrics.json"
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    # Get the existing SVD score to calculate improvement
    svd_rmse = all_metrics.get("SVD_RMSE", 0.90) 
    svd_ste = all_metrics.get("additional_metrics", {}).get("svd", {}).get("ste", 0.0015)

    improvement = ((svd_rmse - pmf_rmse) / svd_rmse) * 100
    winner = "PMF" if pmf_rmse < svd_rmse else "SVD"

    # Safely update JSON dictionary
    all_metrics["PMF_RMSE"] = pmf_rmse
    all_metrics["PMF_vs_SVD_improvement_%"] = round(improvement, 2)
    all_metrics["pmf_optimized_params"] = {
        "alpha": getattr(model, 'alpha', 1.0),
        "k": model.n_factors,
        "epochs": model.n_epochs,
        "burn_in": getattr(model, 'burn_in', 15)
    }
    if "additional_metrics" not in all_metrics:
        all_metrics["additional_metrics"] = {}
    all_metrics["additional_metrics"]["pmf"] = pmf_metrics

    all_metrics["audit_summary"] = {
        "winner": winner,
        "statistically_significant": bool(abs(pmf_rmse - svd_rmse) > (svd_ste * 2)),
        "target_met": bool(pmf_rmse <= 0.85 and improvement >= 5.0)
    }

    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=4)

    logger.info(f"🏁 Phase 4 Complete. PMF Improvement over SVD: {improvement:.2f}%")
    logger.info(f"Metrics updated in {metrics_file}")

if __name__ == "__main__":
    run_pmf_pipeline()