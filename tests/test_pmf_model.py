import pytest
import numpy as np
import pandas as pd
from models.pmf_model import PMFRecommender


@pytest.fixture
def synthetic_data():
    """Generates a small synthetic pivot table for testing PMF."""
    # Your PMF model expects a pivot-table style DataFrame (users as rows, items as columns)
    data = np.array([[5.0, 4.0, np.nan], [np.nan, 1.0, 2.0], [5.0, np.nan, 1.0]])
    df = pd.DataFrame(data, index=[0, 1, 2], columns=[0, 1, 2])
    return df


def test_pmf_initialization():
    """Verify hyperparameters are set correctly."""
    model = PMFRecommender(n_factors=10, n_epochs=20, burn_in=5)
    assert model.n_factors == 10
    assert model.n_epochs == 20
    assert model.burn_in == 5
    assert len(model.U_samples) == 0


def test_pmf_fit_shape(synthetic_data):
    """Verify that U and V samples have correct dimensions after fitting."""
    df = synthetic_data
    n_users, n_items = df.shape
    factors = 5
    # Use a small epoch count for speed in testing
    model = PMFRecommender(n_factors=factors, n_epochs=5, burn_in=2, thin=1)

    # Corrected: fit only takes the dataframe
    model.fit(df)

    # Samples collected = (epochs - burn_in) / thin
    # Epochs 0,1 are burn-in. Samples collected at 2, 3, 4 = 3 samples.
    assert len(model.U_samples) == 3
    assert model.U_samples[0].shape == (n_users, factors)
    assert model.V_samples[0].shape == (n_items, factors)


def test_pmf_prediction_logic(synthetic_data):
    """Verify that predict() returns valid predictions."""
    df = synthetic_data
    model = PMFRecommender(n_factors=2, n_epochs=4, burn_in=1)
    model.fit(df)

    # Predict for specific pairs
    test_users = np.array([0, 1])
    test_items = np.array([0, 2])
    preds = model.predict(test_users, test_items)

    assert len(preds) == 2
    assert isinstance(preds, np.ndarray)
    assert np.all(np.isfinite(preds))


def test_pmf_not_fitted_error():
    """Ensures predict() raises RuntimeError if called before fit()."""
    model = PMFRecommender()
    # Matches your exact code message: "Model not fitted yet. Call fit() first."
    with pytest.raises(RuntimeError, match="Model not fitted yet"):
        model.predict(np.array([0]), np.array([0]))


def test_pmf_reproducibility(synthetic_data):
    """Verify that random_state ensures consistent results."""
    df = synthetic_data

    model1 = PMFRecommender(n_factors=2, n_epochs=3, burn_in=1, random_state=42)
    model1.fit(df)
    res1 = model1.predict(np.array([0]), np.array([1]))

    model2 = PMFRecommender(n_factors=2, n_epochs=3, burn_in=1, random_state=42)
    model2.fit(df)
    res2 = model2.predict(np.array([0]), np.array([1]))

    assert np.allclose(res1, res2)
