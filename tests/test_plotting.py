import pandas as pd
from src.plotting import Plotter
from src.backtester import Backtester


def test_plotter_runs_without_error():
    """
    A simple "smoke test" to ensure the plotting function can be called
    without raising an exception.
    """
    # 1. Setup: Create a realistic dataframe that the plotter expects
    data = {
        "Open": [100, 110],
        "High": [110, 120],
        "Low": [90, 100],
        "Close": [110, 120],
        "ret": [0.1, 0.09],
        "signal": [0, 1],
    }
    df = pd.DataFrame(data)

    # Run it through the backtester to get the required columns
    backtester = Backtester()
    backtest_results = backtester.backtest(df)

    # 2. Action & Assertion
    plotter = Plotter()
    # We can't easily test the visual output, so we just check if it runs.
    # In a real application, you might mock fig.show() to prevent plots
    # from appearing during tests.
    plotter.plot_results(backtest_results, n_states=2)
    # If no exception is raised, the test passes."
