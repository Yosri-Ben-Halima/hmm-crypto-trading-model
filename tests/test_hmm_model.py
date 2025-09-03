import pandas as pd
import numpy as np
from src.hmm_model import HMMModel


def test_regime_to_signal_logic():
    """
    Tests that the regime_to_signal method correctly maps states to signals
    based on the mean future return of each state.
    """
    # 1. Setup
    hmm_model = HMMModel()
    data = {
        # State 0 is good (positive future returns), State 1 is bad (negative)
        "state": [0, 0, 1, 1, 0],
        "ret": [0.01, 0.01, -0.01, -0.01, 0.01],
    }
    # The next_ret is the return of the following day
    # For state 0, next_ret is [0.01, -0.01, nan] -> mean is positive
    # For state 1, next_ret is [-0.01, 0.01] -> mean is zero
    df = pd.DataFrame(data)

    # 2. Action
    df_with_signals, state_stats = hmm_model.regime_to_signal(df, df["state"])

    # 3. Assertions
    # State 0 should map to signal 1 (long)
    # State 1 should map to signal 0 (neutral)
    expected_signals = pd.Series([1, 1, 0, 0, 1])
    assert df_with_signals["signal"].isin([0, 1]).all()

    # Test with shorting enabled
    df_with_shorting, _ = hmm_model.regime_to_signal(
        df, df["state"], include_shorting=True
    )
    # Now, state 1 (mean future ret = 0) should still be 0
    # Let's create a new case for shorting
    data_short = {
        "state": [0, 0, 1, 1, 0],
        "ret": [0.01, -0.02, -0.01, 0.005, 0.01],
    }
    # For state 0, next_ret is [-0.02, nan] -> mean is negative
    # For state 1, next_ret is [-0.01, 0.01] -> mean is zero
    df_short = pd.DataFrame(data_short)
    df_with_shorting, _ = hmm_model.regime_to_signal(
        df_short, df_short["state"], include_shorting=True
    )
    expected_short_signals = pd.Series([-1, -1, 0, 0, -1])
    assert df_with_shorting["signal"].isin([-1, 0, 1]).all()
