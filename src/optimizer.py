import pandas as pd
from .config import SEED
from .hmm_model import HMMModel
from .backtester import Backtester
import plotly.graph_objects as go
from utils.logger import get_logger

logger = get_logger(__name__)


class HMMStateOptimizer:
    def __init__(self, states_range: range, random_state: int = SEED):
        self.states_range = states_range
        self.random_state = random_state
        self.__optimization_results_ = None

    def run_optimization(self, df_features, features, verbose=False):
        """
        Runs the optimization process to find the best number of HMM states.
        """
        logger.info("Optimizing for Number of States...")
        results = []

        def calculate_objective(df: pd.DataFrame):
            """
            Calculates a custom objective score.
            Rewards outperforming returns and penalizes underperforming returns.
            """
            daily_performance_diff = df["strategy_ret"] - df["ret"]
            outperforming_returns = daily_performance_diff[
                daily_performance_diff > 0
            ].sum()
            underperforming_returns = daily_performance_diff[
                daily_performance_diff < 0
            ].sum()

            objective_score = outperforming_returns**2 - underperforming_returns**2
            return -1 * objective_score

        for n_states in self.states_range:
            if verbose:
                print(f"Testing {n_states} states...")

            # 3. Fit HMM
            hmm_model = HMMModel(n_states=n_states, random_state=self.random_state)
            hidden_states = hmm_model.fit(features, verbose=False)

            # 4. Generate Signals
            df_with_signals, _ = hmm_model.regime_to_signal(
                df_features.copy(), hidden_states, verbose=False
            )

            # 5. Backtest
            backtester = Backtester()
            backtest_results = backtester.backtest(df_with_signals, verbose=False)

            # 6. Calculate objective score
            score = calculate_objective(backtest_results)

            results.append({"n_states": n_states, "score": score})
            if verbose:
                print(f"  Score: {score:.4f}")

        # Find the best result
        best_result = min(results, key=lambda x: x["score"])
        if verbose:
            logger.info(
                f"Best number of states: {best_result['n_states']} with score {best_result['score']:.4f}"
            )
        self.__optimization_results_ = pd.DataFrame(results)

        return best_result["n_states"], best_result["score"]

    def plot_optimization_results(self, best_n_states, best_score):
        if self.__optimization_results_ is not None:
            df = self.__optimization_results_

            fig = go.Figure()

            # Line for all results
            fig.add_trace(
                go.Scatter(
                    x=df["n_states"],
                    y=df["score"],
                    mode="lines+markers",
                    name="Scores",
                    line=dict(color="blue"),
                    marker=dict(size=6),
                )
            )

            # Highlight best point
            fig.add_trace(
                go.Scatter(
                    x=[best_n_states],
                    y=[best_score],
                    mode="markers+text",
                    name="Best",
                    text=[f"Best: {best_score:.4f}"],
                    textposition="top center",
                    marker=dict(size=12, color="red", symbol="star"),
                )
            )

            # Layout formatting
            fig.update_layout(
                title="HMM States Optimization",
                xaxis_title="Number of States",
                yaxis_title="Score",
                template="plotly_dark",
                height=500,
                margin=dict(l=50, r=50, t=50, b=50),
            )

            fig.show()

    @property
    def optimization_results(self):
        if self.__optimization_results_ is not None:
            return self.__optimization_results_
