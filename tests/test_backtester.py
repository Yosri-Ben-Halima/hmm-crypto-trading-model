import pandas as pd
import numpy as np
from src.backtester import Backtester


def test_transaction_costs():
    """
    Tests that transaction costs (commission + slippage) are correctly deducted
    for both simple trades (size=1) and position reversals (size=2).
    """
    # 1. Setup: Configure a backtester with high, easy-to-calculate costs
    commission = 0.01  # 1%
    slippage = 0.01  # 1%
    # total_cost_per_trade = commission + slippage  # 0.02

    backtester = Backtester(commission=commission, slippage=slippage)

    # Create a dummy dataframe with signals that will generate
    # a simple trade and a reversal trade.
    data = {
        "ret": [0.1, 0.1, 0.1, 0.1, 0.1],  # Daily returns are all 10%
        "signal": [0, 1, 1, -1, 0],  # Signals for the *next* day
    }
    df = pd.DataFrame(data)

    # Based on the signals, the backtest will calculate these positions and trades:
    # position: [0,   0,   1,   1,  -1]  (signal shifted by 1)
    # trade:    [0,   0,   1,   0,   2]  (abs diff of position)

    # 2. Action: Run the backtest
    results = backtester.backtest(df)

    # 3. Assertions: Check if the strategy returns are correct after costs

    # The backtest drops the first row due to NaN in `trade`, so iloc is offset by 1
    # from the original df.

    # On Day 2 (original index 1), position is 0, so strategy return should be 0
    expected_ret_day2 = 0.0
    assert np.isclose(results["strategy_ret"].iloc[0], expected_ret_day2)

    # On Day 3 (original index 2), we enter a long position (trade=1).
    # Expected return = (position * ret) - (trade_size * cost)
    # Expected return = (1 * 0.1) - (1 * 0.02) = 0.08
    expected_ret_day3 = 0.08
    assert np.isclose(results["strategy_ret"].iloc[1], expected_ret_day3)

    # On Day 4 (original index 3), we hold the position (trade=0).
    # Expected return = (position * ret) - (trade_size * cost)
    # Expected return = (1 * 0.1) - (0 * 0.02) = 0.1
    expected_ret_day4 = 0.1
    assert np.isclose(results["strategy_ret"].iloc[2], expected_ret_day4)

    # On Day 5 (original index 4), we reverse to a short position (trade=2).
    # Expected return = (position * ret) - (trade_size * cost)
    # Expected return = (-1 * 0.1) - (2 * 0.02) = -0.1 - 0.04 = -0.14
    expected_ret_day5 = -0.14
    assert np.isclose(results["strategy_ret"].iloc[3], expected_ret_day5)
