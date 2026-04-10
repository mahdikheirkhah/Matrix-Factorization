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
    actuals = np.array(actuals)
    preds = np.array(preds)
    
    if len(actuals) == 0:
        return {"rmse": 999, "mae": 999, "auc": 0, "precision": 0, "recall": 0, "ste": 0}

    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae = mean_absolute_error(actuals, preds)
    y_true_binary = (actuals >= threshold).astype(int)
    
    # Probabilistic AUC needs the raw predictions
    try:
        auc = roc_auc_score(y_true_binary, preds)
    except:
        auc = 0.5
        
    y_pred_binary = (preds >= threshold).astype(int)
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    
    errors = np.abs(actuals - preds)
    ste = np.std(errors) / np.sqrt(len(errors))
    
    return {
        "rmse": float(rmse), "mae": float(mae), "auc": float(auc),
        "precision": float(precision), "recall": float(recall), "ste": float(ste)
    }

def run_svd_hpo(train_matrix, test_df, user_means, user_map, movie_map):
    ks = [21]
    best_rmse = float('inf')
    best_params, best_metrics = {}, {}

    try:
        for k in ks:
            logger.info(f"🔎 SVD Tuning | testing k={k}...")
            model = SVDRecommender(k=k) 
            
            # CRITICAL: Ensure SVDRecommender.fit returns a DataFrame
            preds_df = model.fit(train_matrix)
            preds_df.columns = preds_df.columns.astype(np.int64)
            print(f"Index Type: {type(preds_df.index[0])}")
            print(f"Column Type: {type(preds_df.columns[0])}")
            # If it returned a numpy array, convert it back to DataFrame to match the pipeline
            if isinstance(preds_df, np.ndarray):
                preds_df = pd.DataFrame(preds_df, index=train_matrix.index, columns=train_matrix.columns)

            actuals, predictions = [], []
            for _, row in test_df.iterrows():
                u_id, m_id, actual = int(row["user_id"]), int(row["movie_id"]), row["rating"]
                
                # Check if User exists in training
                if u_id in train_matrix.index:
                    u_idx = user_map[u_id]
                    
                    # Safe lookup in the residuals DataFrame
                    try:
                        residual = preds_df.at[u_id, m_id] if m_id in preds_df.columns else 0.0
                    except:
                        residual = 0.0
                    if(residual != 0.0):
                        print(f"residual: {residual:.10f}")    
                    p = np.clip(residual + user_means[u_idx], 1, 5)
                    actuals.append(actual)
                    predictions.append(p)

            metrics = evaluate_metrics(actuals, predictions)
            logger.info(f"RMSE: {metrics['rmse']}")
            if metrics['rmse'] < best_rmse:
                best_rmse = metrics['rmse']
                best_metrics = metrics
                best_params = {"k": k}
                
                os.makedirs("reports", exist_ok=True)
                np.save("reports/svd_predictions_best.npy", preds_df.values)

        return best_params, best_metrics
        
    except Exception as e:
        logger.error(f"❌ Error in SVD HPO: {e}", exc_info=True)
        return None, None # Return None to indicate failure

def run_pmf_hpo(train_matrix, test_df, user_means, user_map, movie_map):
    try:
        logger.info(f"🔎 PMF Tuning | Running Bayesian MCMC Ensemble...")
        model = PMFRecommender(
            n_factors=24,
            n_epochs=100,
            burn_in=20,
            thin=2,
            a0=20.0, 
            b0=1.0,
            alpha_init= 20.5,
            alpha_u_init= 20.0,
            alpha_v_init= 15.0,
            validation_split=0.15
        )
        
        # Fit returns the Ensembled DataFrame
        preds_df = model.fit(train_matrix) 
        preds_df.columns = preds_df.columns.astype(np.int64)
        actuals, predictions = [], []
        for _, row in test_df.iterrows():
            u_id, m_id, actual = int(row["user_id"]), int(row["movie_id"]), row["rating"]
            
            if u_id in train_matrix.index:
                u_idx = user_map[u_id]
                
                try:
                    residual = preds_df.at[u_id, m_id] if m_id in preds_df.columns else 0.0
                except:
                    residual = 0.0
                    
                p = np.clip(residual + user_means[u_idx], 1, 5)
                actuals.append(actual)
                predictions.append(p)

        metrics = evaluate_metrics(actuals, predictions)
        
        os.makedirs("reports/pmf_factors", exist_ok=True)
        np.save("reports/pmf_factors/U_factors_best.npy", model.U)
        np.save("reports/pmf_factors/V_factors_best.npy", model.V)
        logger.info(f"🏆 PMF Final Test RMSE: {metrics['rmse']:.4f}")
        return {"factors": model.n_factors}, metrics
        
    except Exception as e:
        logger.error(f"❌ Error in PMF HPO: {e}", exc_info=True)
        return None, None

def run_comprehensive_audit():
    loader = MovieLensLoader()
    train_matrix = loader.load_user_item_matrix()
    print(train_matrix.shape)
    test_df = pd.read_csv("processed/test_ratings.csv")
    user_means = np.load("processed/user_means.npy")
    
    user_map = {id: i for i, id in enumerate(train_matrix.index)}
    movie_map = {id: i for i, id in enumerate(train_matrix.columns.astype(int))}

    # 1. Evaluate Models
    svd_params, svd_metrics = run_svd_hpo(train_matrix, test_df, user_means, user_map, movie_map)
    pmf_params, pmf_metrics = run_pmf_hpo(train_matrix, test_df, user_means, user_map, movie_map)
    
    # 2. FIX: Check for failures before calculating improvement
    if svd_metrics is None or pmf_metrics is None:
        logger.error("🛑 One of the models failed. Improvement cannot be calculated. Check logs above.")
        return

    # 3. Results Comparison
    improvement = ((svd_metrics['rmse'] - pmf_metrics['rmse']) / svd_metrics['rmse']) * 100
    winner = "PMF" if pmf_metrics['rmse'] < svd_metrics['rmse'] else "SVD"
    final_report = {
        "SVD_RMSE": svd_metrics['rmse'],
        "PMF_RMSE": pmf_metrics['rmse'],
        "PMF_vs_SVD_improvement_%": round(improvement, 2),
        "svd_optimized_params": svd_params,
        "pmf_optimized_params": pmf_params,
        "additional_metrics": {"svd": svd_metrics, "pmf": pmf_metrics},
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