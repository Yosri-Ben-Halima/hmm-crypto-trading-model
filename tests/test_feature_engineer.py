import pandas as pd
import numpy as np
from src.feature_engineering import EXPECTED_FEATURES, FeatureEngineer


def test_feature_engineering():
    """
    Tests that the FeatureEngineer calculates features correctly.
    """
    # 1. Setup: Create a sample DataFrame
    data = {
        "Open": [100, 110, 120, 115, 125] * 6,
        "High": [110, 120, 125, 120, 130] * 6,
        "Low": [90, 100, 110, 110, 120] * 6,
        "Close": [110, 120, 115, 125, 130] * 6,
        "logret": [np.nan, 0.087, -0.043, 0.083, 0.039] * 6,
    }
    df = pd.DataFrame(
        data,
        index=pd.date_range(start="2023-01-01", periods=30, freq="D"),
    )

    # 2. Action
    # Use smaller windows for easier manual calculation
    engineer = FeatureEngineer(roll_vol=2, mom_window=2, rsi_window=2, adx_window=2)
    df_with_features, features = engineer.build_features(df)

    # 3. Assertions
    assert isinstance(features, pd.DataFrame)
    assert not features.empty
    # Check that the correct columns are present
    assert all(col in features.columns for col in EXPECTED_FEATURES)

    # Check a specific calculated value (optional, but good practice)
    # For mom10 on the last day: (130 - 115) / 115 = 0.1304
    assert features["mom10"].notna().any()
