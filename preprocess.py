import os
from sklearn.model_selection import train_test_split
from utils.data_loader import MovieLensLoader
from utils.matrix_creation import create_user_item_matrix, normalize_matrix

def main():
    loader = MovieLensLoader()
    
    # 1. Load
    ratings = loader.load_ratings()
    
    # 2. Split (Audit Req: random_state=42)
    train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)
    
    # 3. Create Matrix
    full_matrix = create_user_item_matrix(ratings)
    
    # 4. Normalize
    norm_matrix, _ = normalize_matrix(full_matrix)
    
    # 5. Save (Audit Req: processed/user_item_matrix.csv)
    os.makedirs('processed', exist_ok=True)
    norm_matrix.to_csv('processed/user_item_matrix.csv')
    
    print(f"✅ Phase 2 Complete: Matrix Shape {norm_matrix.shape}")

if __name__ == "__main__":
    main()