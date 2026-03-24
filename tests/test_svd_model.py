import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from models.svd_model import SVDRecommender

def test_svd_reconstruction_shape():
    """Verify that the predicted matrix matches the input matrix shape."""
    data = np.random.rand(10, 10)
    df = pd.DataFrame(data)
    model = SVDRecommender(k=2)
    preds = model.fit(df)
    
    assert preds.shape == (10, 10)
    assert not np.isnan(preds).any()

def test_svd_fit_failure():
    """Verify error handling for invalid k (k >= dimensions)."""
    df = pd.DataFrame(np.random.rand(5, 5))
    model = SVDRecommender(k=10) 
    with pytest.raises(Exception):
        model.fit(df)

def test_svd_with_all_zeros():
    """Flow: Matrix is completely empty/zero. Should not crash."""
    df = pd.DataFrame(np.zeros((10, 10)))
    model = SVDRecommender(k=2)
    preds = model.fit(df)
    
    # Reconstructed matrix from zeros should be all zeros
    assert np.all(preds == 0)
    assert preds.shape == (10, 10)

def test_svd_with_single_row_or_column():
    """Flow: Very small matrix. Linear algebra solvers often struggle here."""
    # Create a 3x10 matrix (Users < k)
    df = pd.DataFrame(np.random.rand(3, 10))
    model = SVDRecommender(k=2) # k must be < min(shape)
    preds = model.fit(df)
    
    assert preds.shape == (3, 10)

def test_svd_reproducibility():
    """Flow: Ensure that fitting twice on the same data yields same results."""
    data = np.random.rand(10, 10)
    df = pd.DataFrame(data)
    
    model1 = SVDRecommender(k=3)
    preds1 = model1.fit(df)
    
    model2 = SVDRecommender(k=3)
    preds2 = model2.fit(df)
    
    # Check if they are almost equal (floating point tolerance)
    np.testing.assert_array_almost_equal(preds1, preds2, decimal=5)

def test_svd_sparse_input_handling():
    """Flow: Ensure the model handles DataFrames with many NaNs (converted to 0)."""
    data = [[5, np.nan, 0], [np.nan, 4, np.nan], [0, 0, 2]]
    df = pd.DataFrame(data).fillna(0)
    
    model = SVDRecommender(k=1)
    preds = model.fit(df)
    
    assert preds.shape == (3, 3)
    assert not np.any(np.isnan(preds))