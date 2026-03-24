import pytest
import pandas as pd
import numpy as np
from utils.matrix_creation import create_user_item_matrix, normalize_matrix


@pytest.fixture
def sample_ratings():
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "movie_id": [101, 102, 101, 103],
            "rating": [5, 3, 4, 2],
        }
    )


def test_create_user_item_matrix_logic(sample_ratings):
    """Flow: Verify pivot table shapes and fillna(0) logic."""
    matrix = create_user_item_matrix(sample_ratings)

    # Check dimensions: 2 users, 3 unique movies
    assert matrix.shape == (2, 3)
    # Check that movie 102 for user 2 is 0 (it was missing in sample)
    assert matrix.loc[2, 102] == 0


def test_normalize_matrix_math():
    """Flow: Verify that row means are zero after normalization."""
    # Create a simple 2x2 matrix
    matrix = pd.DataFrame([[5, 1], [4, 2]], index=[1, 2], columns=[10, 20])

    norm_df, means = normalize_matrix(matrix)

    # After subtracting the mean, the sum of each row should be effectively 0
    row_sums = norm_df.sum(axis=1)
    np.testing.assert_allclose(row_sums, 0, atol=1e-7)
    # Check that means were calculated correctly: (5+1)/2 = 3
    np.testing.assert_allclose(means[0], 3.0, atol=1e-7)


def test_create_matrix_missing_columns():
    """Flow: Dataframe has wrong columns (Expected KeyError)."""
    bad_df = pd.DataFrame({"wrong_col": [1, 2]})
    with pytest.raises(KeyError):
        create_user_item_matrix(bad_df)


def test_filter_sparse_data_logic():
    """Flow: Verify that users/movies with low counts are dropped."""
    from utils.matrix_creation import filter_sparse_data

    # Create data where Movie 99 has only 1 rating, and User 9 has only 1
    df = pd.DataFrame(
        {"user_id": [1, 1, 1, 9], "movie_id": [10, 20, 30, 99], "rating": [5, 5, 5, 1]}
    )

    # Set min_ratings to 2. This should drop User 9 and Movie 99.
    filtered_df = filter_sparse_data(
        df, min_ratings_per_user=2, min_ratings_per_movie=1
    )

    assert 9 not in filtered_df["user_id"].values
    assert 99 not in filtered_df["movie_id"].values
    assert len(filtered_df) == 3  # Only User 1's ratings remain
