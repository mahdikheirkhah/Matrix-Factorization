import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from utils.data_loader import MovieLensLoader

@pytest.fixture
def loader():
    return MovieLensLoader(data_path="fake_data")

def test_load_ratings_success(loader):
    """Flow: File exists and loads correctly."""
    # Mocking read_csv to avoid needing a real .dat file
    mock_data = pd.DataFrame({
        'user_id': [1, 2],
        'movie_id': [10, 20],
        'rating': [5.0, 3.0],
        'timestamp': [999, 888]
    })
    
    with patch("pandas.read_csv", return_value=mock_data):
        df = loader.load_ratings()
        assert len(df) == 2
        assert list(df.columns) == ['user_id', 'movie_id', 'rating', 'timestamp']

def test_load_ratings_file_not_found(loader):
    """Flow: File does not exist (Expected Exception)."""
    with patch("pandas.read_csv", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            loader.load_ratings()

def test_load_ratings_empty_file(loader):
    """Flow: File is empty (Expected Warning/Empty DF)."""
    with patch("pandas.read_csv", return_value=pd.DataFrame()):
        df = loader.load_ratings()
        assert df.empty