# 🎬 Matrix Factorization Recommender System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Scipy](https://img.shields.io/badge/Scipy-1.11%2B-lightgrey)

## 📌 Overview
This project implements an enterprise-grade movie recommendation engine built to compete with industry giants. It utilizes advanced collaborative filtering techniques—specifically **Singular Value Decomposition (SVD)** and **Probabilistic Matrix Factorization (PMF)**—to learn latent user preferences and generate highly personalized movie recommendations.

The project includes a fully interactive **Streamlit dashboard** allowing product teams to input User IDs, view real-time recommendations, and compare the performance of the SVD and PMF models. It is highly robust, featuring cascading fallback logic to handle "Cold Start" users using global and movie-specific averages.

---

## 🚀 Features
* **Dual-Model Pipeline:** Compares standard SVD (via `scipy.sparse.linalg.svds`) with a custom PMF implementation.
* **Cold-Start Resiliency:** Gracefully handles new users and unseen data without crashing, falling back to smart baseline averages.
* **Interactive Dashboard:** A dynamic UI built in Streamlit for real-time model inference and visualization.
* **Anti-Overfitting:** Implements normalization, sparsity filtering, and latent factor tuning to ensure generalization.

---

## 🛠️ Installation & Setup

This project uses **Poetry** for dependency management to ensure a reproducible environment.

**1. Clone the repository and navigate to the directory:**
```bash
git clone https://github.com/mahdikheirkhah/Matrix-Factorization.git
cd Matrix-Factorization
```

**2. Install dependencies via Poetry:**
```bash
poetry install
```
*(Note: You can also find the exported `requirements.txt` in the root directory if you prefer standard pip).*

**3. Run the complete pipeline (Preprocessing, Tuning, and Audit Export):**
```bash
poetry run python -m scripts.preprocess
poetry run python -m scripts.hypertunning
poetry run python -m scripts.interpretability
poetry run python -m generate_audit_samples
```

**4. Launch the Interactive Dashboard:**
```bash
poetry run streamlit run app.py
```

---

## 🧠 Model Interpretability & Audit Analysis

Matrix Factorization is often viewed as a "black box," but our pipeline allows us to map the math back to human behavior. 

### SVD vs. PMF
While **SVD** decomposes a matrix purely algebraically, **PMF** frames the problem probabilistically, assuming ratings are influenced by Gaussian noise. This allows PMF to scale better with highly sparse datasets like MovieLens, handling missing values organically rather than requiring artificial zero-imputation. In our tests, PMF consistently outperformed SVD, achieving an RMSE below 0.85.

### Global Interpretability (The Latent Factors)
By analyzing the movie weight matrix ($V$), we can extract the real-world meaning of our latent factors. For example, sorting the $V$ matrix reveals that movies with the highest weights in "Factor 0" frequently correspond to the Sci-Fi/Action genre, while "Factor 1" heavily groups 90s Romance films. This demonstrates that the model organically learned genre and style groupings purely from user behavior, without relying on explicit metadata.

### Local Interpretability (Why this specific movie?)
To understand why a specific user receives a certain recommendation, we calculate the element-wise product of their User Vector ($U$) and the Movie Vector ($V$). For instance, if User 5441 is recommended *The Godfather*, it is because both the user's vector and the movie's vector share extremely high weights in the specific latent factor corresponding to Crime/Drama. 

### Recommendation Accuracy: Dense vs. Sparse Users
During evaluation, we analyzed two users from the training set with varying historical data. 
* **The Dense User:** User A had hundreds of historical ratings. Because of this rich data footprint, the PMF model was able to accurately pinpoint their specific latent preferences, resulting in highly personalized and accurate recommendations.
* **The Sparse User:** User B only had roughly 20 ratings. Because User B provided so little data, the model's confidence in their latent features was lower. To compensate, the algorithm relied more heavily on the global movie averages, resulting in recommendations that were slightly less personalized and trended closer to general popularity.

---

## 📁 Project Structure
```text
matrix-factorization-project/
│
├── data/                  # Raw MovieLens dataset files (users, movies, ratings)
├── processed/             # Cleaned matrices, test sets, and user means
├── utils/                 # Core logic (data_loader.py, matrix_creation.py, recommendation.py)
├── reports/               # Audit CSVs, visual plots, and metrics JSON
│   └── pmf_factors/       # Saved numpy arrays for the U and V matrices
├── app.py                 # Streamlit dashboard application
├── generate_audit_samples.py # Script to fulfill auditor export requirements
├── Movie_Recommender_System.ipynb # Exploratory Data Analysis (EDA) & Visualizations
└── README.md
