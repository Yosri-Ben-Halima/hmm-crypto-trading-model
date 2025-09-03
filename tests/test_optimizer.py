import pytest
import pandas as pd
from src.optimizer import HMMStateOptimizer
from src.feature_engineering import FeatureEngineer


@pytest.mark.slow
def test_optimizer_runs():
    """
    A simple integration test to ensure the optimizer runs without errors
    and returns a result in the expected format.
    This test can be slow.
    """
    # 1. Setup: Create a small, realistic dataset
    data = {
        "Open": [100, 110, 120, 115, 125, 130, 140, 135, 145, 150] * 10,
        "High": [110, 120, 125, 120, 130, 140, 145, 140, 150, 155] * 10,
        "Low": [90, 100, 110, 110, 120, 125, 130, 130, 140, 145] * 10,
        "Close": [110, 120, 115, 125, 130, 140, 135, 145, 150, 152] * 10,
        "logret": [0.0, 0.087, -0.043, 0.083, 0.039, 0.074, -0.036, 0.071, 0.034, 0.013]
        * 10,
    }
    df = pd.DataFrame(data)

    engineer = FeatureEngineer()
    df_with_features, features = engineer.build_features(df)

    # 2. Action
    optimizer = HMMStateOptimizer(states_range=range(2, 4))  # Test a small range
    best_n_states, best_score = optimizer.run_optimization(df_with_features, features)

    # 3. Assertions
    assert isinstance(best_n_states, int)
    assert isinstance(best_score, float)
    assert best_n_states > 1
    assert optimizer.optimization_results is not None
    assert not optimizer.optimization_results.empty
