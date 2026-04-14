import pandas as pd
import numpy as np
from utils.data_loader import MovieLensLoader
from utils.recommendation import generate_recommendations


def export_audit_samples():
    loader = MovieLensLoader()
    U, V = loader.load_pmf_factors()
    matrix = loader.load_user_item_matrix()
    movies_df = loader.load_movies()
    user_map = {uid: i for i, uid in enumerate(matrix.index)}

    # Audit Target User IDs
    target_users = [1, 50, 100]

    for uid in target_users:
        recs = generate_recommendations(uid, U, V, user_map, None, movies_df, matrix)
        filename = f"reports/recommendations_user_{uid}.csv"
        recs.to_csv(filename, index=False)
        print(f"✅ Exported audit sample: {filename}")


if __name__ == "__main__":
    export_audit_samples()
