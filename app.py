import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import MovieLensLoader
from utils.recommendation import generate_recommendations

st.set_page_config(page_title="Matrix Factorization Audit Dashboard", layout="wide")

@st.cache_resource
def load_all_assets():
    loader = MovieLensLoader()
    U, V = loader.load_pmf_factors()
    matrix = loader.load_user_item_matrix()
    movies_df = loader.load_movies()
    metrics = loader.load_metrics()
    user_map = {uid: i for i, uid in enumerate(matrix.index)}
    return U, V, matrix, movies_df, metrics, user_map

U, V, matrix, movies_df, metrics, user_map = load_all_assets()

st.title("🎬 Recommendation System Audit Dashboard")

# --- Sidebar: User Selection ---
st.sidebar.header("User Parameters")
user_input = st.sidebar.number_input("Enter User ID:", min_value=1, value=1, step=1)

# --- Main Logic ---
if user_input not in user_map:
    st.error(f"❌ User ID {user_input} not found in the training data.")
else:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(f"Top 10 Recommendations for User {user_input}")
        recs = generate_recommendations(user_input, U, V, user_map, None, movies_df, matrix)
        st.table(recs[['title', 'genres', 'predicted_score']])

    with col2:
        st.subheader("Model Comparison (Audit View)")
        svd_rmse = metrics['additional_metrics']['svd']['rmse']
        pmf_rmse = metrics['additional_metrics']['pmf']['rmse']
        
        # Simple Comparison Chart
        fig, ax = plt.subplots()
        ax.bar(['SVD', 'PMF'], [svd_rmse, pmf_rmse], color=['gray', 'green'])
        ax.set_ylabel("RMSE")
        ax.set_title("Model Error Comparison")
        st.pyplot(fig)
        
        st.metric("Improvement", f"{metrics['PMF_vs_SVD_improvement_%']}%", delta="PMF Winner")