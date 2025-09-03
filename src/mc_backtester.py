import numpy as np
import pandas as pd
from src.backtester import Backtester
from src.hmm_model import HMMModel
from scipy.stats import gaussian_kde
from utils.logger import get_logger

logger = get_logger(__name__)


class MCBacktester:
    """
    Monte Carlo backtester for HMM-based trading strategy.
    """

    def __init__(
        self,
        features_train: pd.DataFrame,
        features_test: pd.DataFrame,
        test_df: pd.DataFrame,
        n_states: int = 14,
        runs: int = 100,
    ):
        """
        Parameters
        ----------
        hmm_model_cls : class
            HMM model class (dependency injection).
        backtester_cls : class
            Backtester class (dependency injection).
        features : pd.DataFrame
            Feature matrix.
        train_df : pd.DataFrame
            Training dataset (with OHLCV + signals).
        test_df : pd.DataFrame
            Test dataset (with OHLCV + signals).
        n_states : int
            Number of hidden states for HMM.
        runs : int
            Number of Monte Carlo runs.
        """
        self.features_train = features_train
        self.features_test = features_test

        self.test_df = test_df
        self.n_states = n_states
        self.runs = runs
        self.benchmark_return = None
        self.sf = None
        self.pdf = None
        self.cdf = None

        # results storage
        self.returns = []
        self.sharpes = []
        self.drawdowns = []
        self.trades = []
        self.paths_equity = []

    def run(self, seeded=False, verbose=True):
        """Run Monte Carlo backtest across multiple simulations."""
        i = 0
        seed = 0
        while i < self.runs:
            try:
                hmm_model = HMMModel(
                    n_states=self.n_states, random_state=seed if seeded else None
                )
                hmm_model.fit(self.features_train, verbose=False)
                if (
                    hasattr(hmm_model.model, "monitor_")
                    and not hmm_model.model.monitor_.converged
                ):
                    last_ll = hmm_model.model.monitor_.history[-1]
                    prev_ll = (
                        hmm_model.model.monitor_.history[-2]
                        if len(hmm_model.model.monitor_.history) > 1
                        else last_ll
                    )
                    delta = last_ll - prev_ll
                    raise RuntimeError(
                        f"HMM failed to converge. "
                        f"Last log-likelihood: {last_ll:.4f}, "
                        f"Delta: {delta:.4f}"
                    )
                hidden_states = hmm_model.predict(self.features_test, verbose=False)
                df_with_signals, _ = hmm_model.regime_to_signal(
                    self.test_df, hidden_states, verbose=False
                )

                backtester = Backtester()
                results = backtester.backtest(df_with_signals, verbose=False)
                metrics = backtester.metrics(results, "strategy_equity")

                self.returns.append(metrics["total_return"])
                self.sharpes.append(metrics["annualized_sharpe"])
                self.drawdowns.append(metrics["max_drawdown"])
                self.trades.append(metrics["number_of_trades"])
                self.paths_equity.append(results["strategy_equity"])
                if verbose:
                    logger.info(f"Run {i + 1}/{self.runs}")
                i += 1

            except Exception as e:  # noqa: F841
                # if verbose:
                #     logger.error(f"Error in run {i + 1} : {e}")
                continue

            finally:
                seed += 1

        if not self.paths_equity:
            average_equity = pd.Series(
                np.nan, index=self.test_df.index, name="average_equity"
            )
        else:
            equity_df = pd.concat(self.paths_equity, axis=1)
            average_equity = equity_df.mean(axis=1)
            average_equity.name = "average_equity"

        avg_df = pd.DataFrame(average_equity)
        avg_df["hodl_equity"] = results["hodl_equity"]
        avg_df["Close"] = results["Close"]
        avg_df["Open"] = results["Open"]
        avg_df["High"] = results["High"]
        avg_df["Low"] = results["Low"]
        avg_df["outperforming"] = avg_df["average_equity"] > avg_df["hodl_equity"]
        self.benchmark_return = backtester.metrics(results, col="hodl_equity")[
            "total_return"
        ]
        pdf = gaussian_kde(self.returns)

        def cdf(x):
            return pdf.integrate_box_1d(-np.inf, x)

        def sf(x):
            return 1 - cdf(x)

        self.sf = sf
        self.pdf = pdf
        self.cdf = cdf

        return self.returns, self.sharpes, self.drawdowns, self.trades, avg_df

    def probability_outperformance(self, mult=1):
        """
        Compute probability that strategy outperforms `mult` times the benchmark.

        Parameters
        ----------
        mult : float
            Multiplier for benchmark return to define outperformance threshold.

        Returns
        -------
        prob : float
            Probability of outperforming benchmark.
        """

        return self.sf(mult * self.benchmark_return)

    def summary_statistics(self):
        """Compute mean and stddev of returns, sharpe ratios, and drawdowns."""
        return {
            "average_return": np.mean(self.returns),
            "std_return": np.std(self.returns, ddof=1),
            "average_sharpe": np.mean(self.sharpes),
            "std_sharpe": np.std(self.sharpes, ddof=1),
            "average_max_drawdown": np.mean(self.drawdowns),
            "std_max_drawdown": np.std(self.drawdowns, ddof=1),
        }
