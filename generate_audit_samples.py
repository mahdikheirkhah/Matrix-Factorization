import pandas as pd
import numpy as np
import os
from utils.data_loader import MovieLensLoader
from utils.recommendation import generate_recommendations

def export_audit_samples():
    loader = MovieLensLoader()
    U, V = loader.load_pmf_factors()
    matrix = loader.load_user_item_matrix()
    movies_df = loader.load_movies()
    user_map = {uid: i for i, uid in enumerate(matrix.index)}

    # Calculate fallbacks for the new 'safe' logic
    print("⏳ Calculating global means for fallback logic...")
    global_mean = matrix.stack().mean()
    movie_means = matrix.mean(axis=0).to_dict()

    # Audit Target User IDs
    target_users = [5441, 1446, 28]
    
    os.makedirs("reports", exist_ok=True)

    for uid in target_users:
        # Use the updated function signature with keyword arguments for safety
        recs = generate_recommendations(
            user_id=uid, 
            U=U, 
            V=V, 
            user_map=user_map, 
            movies_df=movies_df, 
            matrix=matrix,
            global_mean=global_mean,
            movie_means=movie_means
        )
        
        filename = f"reports/recommendations_user_{uid}.csv"
        recs.to_csv(filename, index=False)
        print(f"✅ Exported audit sample: {filename}")

if __name__ == "__main__":
    export_audit_samples()