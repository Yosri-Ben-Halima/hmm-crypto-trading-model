import pandas as pd
from src.data_loader import DataLoader


def test_data_loader_processing(mocker):
    """
    Tests that the DataLoader correctly processes a mocked API response.
    """
    # 1. Setup: Create a mock JSON response from the Binance API
    mock_json_response = [
        [
            1672531200000,  # open_time
            "16500.0",
            "16600.0",
            "16400.0",
            "16550.0",  # o, h, l, c
            "1000.0",
            1672617599999,
            "16550000.0",
            100,
            "500.0",
            "8275000.0",
            "0",
        ],
        [
            1672617600000,  # open_time
            "16550.0",
            "16700.0",
            "16500.0",
            "16650.0",  # o, h, l, c
            "1200.0",
            1672703999999,
            "19980000.0",
            120,
            "600.0",
            "9990000.0",
            "0",
        ],
    ]
    # Mock the requests.get call
    mock_get = mocker.patch("requests.get")
    mock_get.return_value.json.return_value = mock_json_response
    mock_get.return_value.raise_for_status.return_value = None

    # 2. Action
    loader = DataLoader()
    df = loader.get_data()

    # 3. Assertions
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape[0] > 0  # After dropping NaN from logret
    assert "logret" in df.columns
    assert pd.api.types.is_float_dtype(df["Close"])
    assert pd.api.types.is_datetime64_any_dtype(df.index)
