import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.data_loader import MovieLensLoader
from utils.recommendation import generate_recommendations

st.set_page_config(page_title="MF Audit Dashboard", layout="wide")

@st.cache_resource
def load_all_assets():
    loader = MovieLensLoader()
    U, V = loader.load_pmf_factors()
    matrix = loader.load_user_item_matrix()
    movies_df = loader.load_movies()
    metrics = loader.load_metrics()
    
    # 1. Get Training Users
    user_map = {uid: i for i, uid in enumerate(matrix.index)}
    
    # 2. Get Test Users
    if os.path.exists("processed/test_ratings.csv"):
        test_df = pd.read_csv("processed/test_ratings.csv")
        test_users = set(test_df['user_id'].unique())
    else:
        test_users = set()

    # 3. Get All Original Users from users.dat
    try:
        # Adjust this path if your raw data is in a different folder (like data/ml-1m/)
        users_file = "data/users.dat" if os.path.exists("data/users.dat") else "data/ml-1m/users.dat"
        users_df = pd.read_csv(users_file, sep='::', engine='python', usecols=[0], names=['user_id'])
        all_users = set(users_df['user_id'].unique())
    except Exception:
        # Fallback if raw file isn't on the server: MovieLens 1M has exactly 6040 users
        all_users = set(range(1, 6041))
    
    # Precalculate fallbacks for safe_predict
    global_mean = matrix.stack().mean()
    movie_means = matrix.mean(axis=0).to_dict()
    
    return U, V, matrix, movies_df, metrics, user_map, test_users, all_users, global_mean, movie_means

# Load everything
(U, V, matrix, movies_df, metrics, 
 user_map, test_users, all_users, 
 global_mean, movie_means) = load_all_assets()

st.title("🎬 Recommendation System Audit Dashboard")

# --- Sidebar ---
st.sidebar.header("User Control Panel")
user_input = st.sidebar.number_input("Enter User ID:", min_value=1, value=1, step=1)

# --- Dynamic User Routing Logic ---
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader(f"Recommendations for User {user_input}")
    
    # Track if we should generate recommendations at all
    is_valid_user = True

    # 1. Check Training Set
    if user_input in user_map:
        st.success(f"✅ **User ID {user_input} was in the training set.** (Showing Personalized PMF Predictions)")
    
    # 2. Check Test Set
    elif user_input in test_users:
        st.info(f"ℹ️ **User ID {user_input} found in the test set.** (Showing Global Fallback Predictions)")
    
    # 3. Check Original Dataset (Sparsity Filtered)
    elif user_input in all_users:
        st.warning(f"⚠️ **Cold Start:** User {user_input} removed during preprocessing due to sparsity. (Showing Global Fallback Predictions)")
    
    # 4. Completely Invalid User
    else:
        st.error(f"❌ **Invalid User ID:** {user_input} was not found in any dataset.")
        is_valid_user = False

    # Generate and show recommendations ONLY if it's a valid ID
    if is_valid_user:
        recs = generate_recommendations(
            user_id=user_input, 
            U=U, V=V, 
            user_map=user_map, 
            movies_df=movies_df, 
            matrix=matrix,
            global_mean=global_mean,
            movie_means=movie_means
        )
        st.dataframe(recs[['title', 'genres', 'predicted_score']].head(10), use_container_width=True)


with col2:
    st.subheader("System Performance (Audit View)")
    svd_rmse = metrics['additional_metrics']['svd']['rmse']
    pmf_rmse = metrics['additional_metrics']['pmf']['rmse']
    
    # Performance Chart
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(['SVD', 'PMF'], [svd_rmse, pmf_rmse], color=['#95a5a6', '#2ecc71'])
    ax.set_ylabel("RMSE")
    ax.set_ylim([0, max(svd_rmse, pmf_rmse) * 1.2])
    st.pyplot(fig)
    
    st.metric("Model Improvement", f"{metrics['PMF_vs_SVD_improvement_%']}%", delta="PMF Winner")
    
    with st.expander("View Full Audit JSON"):
        st.json(metrics)