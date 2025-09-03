import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from src.config import N_STATES


class Plotter:
    def plot_results(self, df: pd.DataFrame, n_states: int = N_STATES):
        # Create a subplot figure with 3 rows
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                "BTC/USDT",
                "Portfolio Value Curves",
            ),
        )

        # --- Row 1: BTC Candlestick ---
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="BTC/USDT",
            ),
            row=1,
            col=1,
        )

        # --- Row 2: Portfolio Value Curves ---
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["strategy_equity"],
                name="HMM Strategy",
                line=dict(color="cyan"),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["hodl_equity"],
                name="HODL",
                line=dict(color="white"),
                opacity=0.7,
            ),
            row=2,
            col=1,
        )

        # --- Highlight periods ---
        change_points = df[df["outperforming"] != df["outperforming"].shift()]

        start_date = df.index[0]
        for end_date in change_points.index:
            is_outperforming = df.loc[start_date, "outperforming"]
            color = "green" if is_outperforming else "red"
            fig.add_vrect(
                x0=start_date,
                x1=end_date,
                fillcolor=color,
                opacity=0.2,
                layer="below",
                line_width=0,
                row=2,
                col=1,
            )
            start_date = end_date

        # Last period
        is_outperforming = df.loc[start_date, "outperforming"]
        color = "green" if is_outperforming else "red"
        fig.add_vrect(
            x0=start_date,
            x1=df.index[-1],
            fillcolor=color,
            opacity=0.2,
            layer="below",
            line_width=0,
            row=2,
            col=1,
        )

        # --- Layout & Formatting ---
        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        fig.update_yaxes(title_text="BTC/USDT", row=1, col=1, tickformat="~s")
        fig.update_yaxes(title_text="Portfolio Value", row=2, col=1, tickformat="~s")

        fig.update_layout(
            title=f"Performance Dashboard of {n_states}-State HMM",
            height=900,
            template="plotly_dark",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            margin=dict(l=50, r=50, t=50, b=50),
            barmode="overlay",
        )

        fig.show()

    def plot_mc_results(self, avg_df: pd.DataFrame, runs: int):
        # Create a subplot figure with 3 rows
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                "BTC/USDT",
                "Portfolio Value Curves",
            ),
        )

        # --- Row 1: BTC Candlestick ---
        fig.add_trace(
            go.Candlestick(
                x=avg_df.index,
                open=avg_df["Open"],
                high=avg_df["High"],
                low=avg_df["Low"],
                close=avg_df["Close"],
                name="BTC/USDT",
            ),
            row=1,
            col=1,
        )

        # --- Row 2: Portfolio Value Curves ---
        fig.add_trace(
            go.Scatter(
                x=avg_df.index,
                y=avg_df["average_equity"],
                name="Average HMM Strategy Path",
                line=dict(color="cyan"),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=avg_df.index,
                y=avg_df["hodl_equity"],
                name="Benchmark (HODL)",
                line=dict(color="white"),
                opacity=0.7,
            ),
            row=2,
            col=1,
        )

        # --- Highlight periods ---
        change_points = avg_df[
            avg_df["outperforming"] != avg_df["outperforming"].shift()
        ]

        start_date = avg_df.index[0]
        for end_date in change_points.index:
            is_outperforming = avg_df.loc[start_date, "outperforming"]
            color = "green" if is_outperforming else "red"
            fig.add_vrect(
                x0=start_date,
                x1=end_date,
                fillcolor=color,
                opacity=0.2,
                layer="below",
                line_width=0,
                row=2,
                col=1,
            )
            start_date = end_date

        # Last period
        is_outperforming = avg_df.loc[start_date, "outperforming"]
        color = "green" if is_outperforming else "red"
        fig.add_vrect(
            x0=start_date,
            x1=avg_df.index[-1],
            fillcolor=color,
            opacity=0.2,
            layer="below",
            line_width=0,
            row=2,
            col=1,
        )

        # --- Layout & Formatting ---
        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        fig.update_yaxes(title_text="BTC/USDT", row=1, col=1, tickformat="~s")
        fig.update_yaxes(title_text="Portfolio Value", row=2, col=1, tickformat="~s")

        fig.update_layout(
            title=f"Performance Dashboard of {runs} Runs of HMM Strategy",
            height=900,
            template="plotly_dark",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            margin=dict(l=50, r=50, t=50, b=50),
            barmode="overlay",
        )

        fig.show()

    def plot_return_distribution(
        self,
        mc_backtester,
        nbinsx: int = None,
    ):
        runs = mc_backtester.runs

        pdf = mc_backtester.pdf
        returns_array = np.array(mc_backtester.returns)
        benchmark_ret = mc_backtester.benchmark_return
        avg_return = returns_array.mean()
        # std_return = returns_array.std(ddof=1)
        x_vals = np.linspace(min(returns_array) - 0.1, max(returns_array) + 0.1, 500)

        # Histogram (density normalized)
        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=returns_array,
                nbinsx=nbinsx,
                histnorm="probability density",  # normalize to density
                name="HMM Returns",
                marker=dict(color="cyan"),
                opacity=0.6,
            )
        )

        # Vertical line: HODL return
        fig.add_trace(
            go.Scatter(
                x=[benchmark_ret, benchmark_ret],
                y=[0, 3],  # Adjust dynamically later if needed
                mode="lines",
                line=dict(color="red", dash="dash"),
                name=f"HODL Return = {benchmark_ret:.2%}",
            )
        )

        # Vertical line: Average HMM return
        fig.add_trace(
            go.Scatter(
                x=[avg_return, avg_return],
                y=[0, 3],  # Adjust dynamically
                mode="lines",
                line=dict(color="green", dash="dash"),
                name=f"Average HMM Return = {avg_return:.2%}",
            )
        )

        # KDE fit overlay
        pdf_vals = pdf(x_vals)
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=pdf_vals,
                mode="lines",
                line=dict(color="white", width=2, dash="dot"),
                name="KDE Fit",
            )
        )
        fig.update_xaxes(tickformat="0%")
        # Layout
        fig.update_layout(
            title=f"Density of Total Returns After {runs} Runs",
            xaxis_title="Total Return (%)",
            yaxis_title="Density",
            bargap=0.05,
            template="plotly_dark",
            legend=dict(x=0.7, y=0.95),
            height=600,
            width=800,
        )

        fig.show()

    def plot_correlation_heatmap(self, corr: pd.DataFrame):
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                colorbar=dict(title="Correlation"),
            )
        )

        for i in range(len(corr)):
            for j in range(len(corr)):
                if i == j:
                    fig.add_annotation(
                        x=corr.columns[j],
                        y=corr.index[i],
                        text=f"{corr.values[i, j]:.2f}",
                        showarrow=False,
                        font=dict(color="white"),
                    )
                else:
                    fig.add_annotation(
                        x=corr.columns[j],
                        y=corr.index[i],
                        text=f"{corr.values[i, j]:.2f}",
                        showarrow=False,
                        font=dict(
                            color="green" if abs(corr.values[i, j]) < 0.5 else "red"
                        ),
                    )

        mapper = {
            "rsi": "RSI",
            "vol21": "21-Day Volatility",
            "mom10": "Momentum",
            "ret": "Daily Returns",
        }
        fig.update_xaxes(
            tickvals=corr.columns, ticktext=[mapper[key] for key in corr.columns]
        )
        fig.update_yaxes(
            tickvals=corr.columns, ticktext=[mapper[key] for key in corr.columns]
        )

        fig.update_layout(
            title="Feature Correlation (Green: Low Correlation, Red: High Correlation)",
            xaxis=dict(tickangle=0),
            yaxis=dict(autorange="reversed", tickangle=0),
            template="plotly_dark",
            height=700,
            width=700,
        )

        fig.show()
