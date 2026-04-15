import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from utils.data_loader import MovieLensLoader
from utils.recommendation import generate_recommendations


def plot_user_comparison(
    user_id, pmf_recs, filename="reports/user_comparison.png", top_n=10
):
    """Fulfills the 'user_comparison.png' and 'Why movies are recommended' audit requirements"""
    plt.figure(figsize=(10, 6))

    # We will plot the predicted scores of the top 10 movies
    movies = pmf_recs["title"].head(top_n)[::-1]  # Reverse for top-down reading
    scores = pmf_recs["predicted_score"].head(top_n)[::-1]

    # Create a clean bar chart
    bars = plt.barh(movies, scores, color="#2ecc71")

    plt.xlabel("Predicted Rating (1-5 Stars)")
    plt.ylabel("Movie Title")
    plt.title(f"Top {top_n} PMF Recommendations for User {user_id}")
    plt.xlim([0, 5.5])

    # Add the score text to the bars
    for bar in bars:
        plt.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.2f}",
            va="center",
        )

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def export_audit_samples():
    loader = MovieLensLoader()
    U, V = loader.load_pmf_factors()
    matrix = loader.load_user_item_matrix()
    movies_df = loader.load_movies()
    user_map = {uid: i for i, uid in enumerate(matrix.index)}

    # # Load user means for denormalization
    # try:
    #     user_means = np.load("processed/user_means.npy")
    # except FileNotFoundError:
    #     print("⚠️ user_means.npy not found, falling back to 0s")
    #     user_means = np.zeros(len(matrix.index))

    print("⏳ Calculating global means for fallback logic...")
    global_mean = matrix.stack().mean()
    movie_means = matrix.mean(axis=0).to_dict()

    # NOTE: Ensure 5441 and 1446 are in training set, and 28 is in test set!
    target_users = [5441, 1446, 28]

    os.makedirs("reports", exist_ok=True)

    for i, uid in enumerate(target_users):
        recs = generate_recommendations(
            user_id=uid,
            U=U,
            V=V,
            user_map=user_map,
            movies_df=movies_df,
            matrix=matrix,
            global_mean=global_mean,
            movie_means=movie_means,
            top_n=20,
        )

        # 1. Save the CSV
        csv_filename = f"reports/user_{uid}_recommendations.csv"
        recs.to_csv(csv_filename, index=False)
        print(f"✅ Exported CSV: {csv_filename}")

        # 2. Save the required Audit Visualizations
        if i == 0:
            # Save the primary user comparison plot for the first user
            plot_user_comparison(
                uid, recs, filename="reports/user_comparison.png", top_n=20
            )
            print("✅ Exported Plot: reports/user_comparison.png")

        # Optional extra: Save individual plots for the other users just in case
        plot_user_comparison(
            uid, recs, filename=f"reports/top_recommendations_user_{uid}.png", top_n=20
        )


if __name__ == "__main__":
    export_audit_samples()
