import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def generate_recommendations(user_id, U, V, user_map, movie_map, movies_df, matrix, top_n=10):
    """
    Generates top-N recommendations for a user by calculating the dot product
    and filtering out already-rated movies.
    """
    if user_id not in user_map:
        return None
    
    u_idx = user_map[user_id]
    
    # 1. Predict all ratings for this user: (1, K) @ (K, Items) = (1, Items)
    preds = np.dot(U[u_idx], V.T)
    
    # 2. Identify movies the user already rated in the training set
    user_row = matrix.loc[user_id]
    already_rated = user_row[user_row.notna()].index.astype(int).tolist()
    
    # 3. Create a Series for predictions
    rec_series = pd.Series(preds, index=matrix.columns.astype(int))
    
    # 4. Filter out already rated movies
    recommendations = rec_series.drop(labels=already_rated, errors='ignore')
    
    # 5. Get Top N and join with movie titles
    top_ids = recommendations.nlargest(top_n).index
    result = movies_df[movies_df['movie_id'].isin(top_ids)].copy()
    
    # Add the predicted score for transparency
    result['predicted_score'] = result['movie_id'].map(recommendations)
    
    return result.sort_values(by='predicted_score', ascending=False)