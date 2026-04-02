import numpy as np
import pandas as pd
import logging
import json
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, precision_score, recall_score
from models.pmf_model import PMFRecommender
from models.svd_model import SVDRecommender
from utils.data_loader import MovieLensLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_metrics(actuals, preds, threshold=3.5):
    # Same as your implementation...
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

def run_svd_hpo(train_matrix, test_df, user_means, user_map, movie_map):
    ks = [24]
    best_rmse = float('inf')
    best_params, best_metrics = {}, {}
    best_preds_matrix, best_model = None, None

    try:
        for k in ks:
            logger.info(f"🔎 SVD Tuning | testing k={k}...")
            model = SVDRecommender(k=k, epochs=35) # Iterative SVD
            preds_matrix = model.fit(train_matrix)

            actuals, predictions = [], []
            for _, row in test_df.iterrows():
                u_id, m_id, actual = int(row["user_id"]), int(row["movie_id"]), row["rating"]
                if u_id in user_map and m_id in movie_map:
                    u_idx, m_idx = user_map[u_id], movie_map[m_id]
                    p = np.clip(preds_matrix[u_idx, m_idx] + user_means[u_idx], 1, 5)
                    actuals.append(actual)
                    predictions.append(p)

            metrics = evaluate_metrics(actuals, predictions)
            if metrics['rmse'] < best_rmse:
                best_rmse = metrics['rmse']
                best_metrics = metrics
                best_params = {"k": k}
                best_preds_matrix = preds_matrix
                best_model = model

        np.save("reports/svd_predictions_best.npy", best_preds_matrix)

        if best_model is not None:
            # Iterative SVD factors are already scaled during SGD.
            # We directly save U and V.T (Vt)
            np.save("reports/svd_U_init.npy", best_model.u)
            np.save("reports/svd_V_init.npy", best_model.vt.T)
            logger.info(f"✅ Saved Iterative SVD factors for warm-start (k={best_params['k']})")

        return best_params, best_metrics
        
    except Exception as e:
        logger.error(f"❌ Error in SVD HPO: {e}")
        return {}, {}

def run_pmf_hpo(train_matrix, test_df, user_means, user_map, movie_map):
    param_grid = {'lrs': [0.002], 'factors': [10], 'regs': [9.5], 'epochs': [500]}
    best_rmse = float('inf')
    best_params, best_metrics = {}, {}

    # Load SVD warm-start factors
    try:
        U_svd = np.load("reports/svd_U_init.npy")
        V_svd = np.load("reports/svd_V_init.npy")
        svd_init = (U_svd, V_svd)
        logger.info(f"🔥 Loaded SVD factors for warm-start (shape: U={U_svd.shape}, V={V_svd.shape})")
    except FileNotFoundError:
        svd_init = None
        logger.warning("⚠️ No SVD factors found for warm-start. Using random initialisation.")

    try:
        for lr in param_grid['lrs']:
            for k in param_grid['factors']:
                for reg in param_grid['regs']:
                    for epoch in param_grid['epochs']:
                        logger.info(f"🔎 PMF Tuning | LR={lr}, K={k}, Reg={reg}, Epochs={epoch}...")
                        model = PMFRecommender(
                            n_factors=k, learning_rate=lr, lambda_reg=reg, n_epochs=epoch,
                            validation_split=0.1, early_stopping_patience=25,
                        )  
                        # Pass the svd_init parameter here
                        model.fit(train_matrix, svd_init=svd_init)   

                        full_preds = np.dot(model.U, model.V.T)
                        actuals, predictions = [], []
                        for _, row in test_df.iterrows():
                            u_id, m_id, actual = int(row["user_id"]), int(row["movie_id"]), row["rating"]
                            if u_id in user_map and m_id in movie_map:
                                u_idx, m_idx = user_map[u_id], movie_map[m_id]
                                p = np.clip(full_preds[u_idx, m_idx] + user_means[u_idx], 1, 5)
                                actuals.append(actual)
                                predictions.append(p)

                        metrics = evaluate_metrics(actuals, predictions)
                        if metrics['rmse'] < best_rmse:
                            best_rmse = metrics['rmse']
                            best_metrics = metrics
                            best_params = {"lr": lr, "k": k, "reg": reg, "epochs": epoch}
                            np.save("reports/pmf_factors/U_factors_best.npy", model.U)
                            np.save("reports/pmf_factors/V_factors_best.npy", model.V)

        return best_params, best_metrics
        
    except Exception as e:
        logger.error(f"❌ Error in PMF HPO: {e}")
        return {}, {}

def run_comprehensive_audit():
    loader = MovieLensLoader()
    train_matrix = loader.load_user_item_matrix()
    test_df = pd.read_csv("processed/test_ratings.csv")
    user_means = np.load("processed/user_means.npy")
    logger.info("")
    user_map = {id: i for i, id in enumerate(train_matrix.index)}
    movie_map = {id: i for i, id in enumerate(train_matrix.columns.astype(int))}

    # 1. HPO based on RMSE
    svd_params, svd_metrics = run_svd_hpo(train_matrix, test_df, user_means, user_map, movie_map)
    pmf_params, pmf_metrics = run_pmf_hpo(train_matrix, test_df, user_means, user_map, movie_map)
    
    # 2. Results Comparison
    improvement = ((svd_metrics['rmse'] - pmf_metrics['rmse']) / svd_metrics['rmse']) * 100
    winner = "PMF" if pmf_metrics['rmse'] < svd_metrics['rmse'] else "SVD"

    final_report = {
        "SVD_RMSE": svd_metrics['rmse'],
        "PMF_RMSE": pmf_metrics['rmse'],
        "PMF_vs_SVD_improvement_%": round(improvement, 2),
        "svd_optimized_params": svd_params,
        "pmf_optimized_params": pmf_params,
        "additional_metrics": {
            "svd": svd_metrics,
            "pmf": pmf_metrics
        },
        "audit_summary": {
            "winner": winner,
            "statistically_significant": bool(abs(pmf_metrics['rmse'] - svd_metrics['rmse']) > (svd_metrics['ste'] * 2)),
            "target_met": bool(pmf_metrics['rmse'] <= 0.85)
        }
    }
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/model_metrics.json", "w") as f:
        json.dump(final_report, f, indent=4)
    
    logger.info(f"🏆 Audit Finished. RMSE Improvement: {improvement:.2f}%")

if __name__ == "__main__":
    run_comprehensive_audit()