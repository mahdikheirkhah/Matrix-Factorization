import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from models.pmf_model import PMFRecommender

def run_pmf_pipeline():
    # 1. Load Data
    train_matrix = pd.read_csv('processed/user_item_matrix.csv', index_col=0)
    
    # 2. Train Model
    model = PMFRecommender(n_epochs=50)
    history = model.fit(train_matrix)
    
    # 3. Save Convergence Plot (Audit Req: reports/pmf_convergence.png)
    os.makedirs('reports', exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(history, marker='o')
    plt.title('PMF Convergence (MSE over Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.savefig('reports/pmf_convergence.png')
    
    # 4. Export Factors (Audit Req: reports/pmf_factors/)
    factor_path = 'reports/pmf_factors/'
    os.makedirs(factor_path, exist_ok=True)
    np.save(f"{factor_path}U_factors.npy", model.U)
    np.save(f"{factor_path}V_factors.npy", model.V)
    
    print(f"🏁 Phase 4 Complete. Convergence plot saved to {factor_path}")

if __name__ == "__main__":
    run_pmf_pipeline()