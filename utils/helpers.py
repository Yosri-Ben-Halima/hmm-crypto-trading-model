from typing import Dict
import pandas as pd

# from src.mc_backtester import MCBacktester


def parse_metrics_str(metrics):
    return (
        " | ".join(
            "{}: {}".format(
                k.replace("_", " ").title(),
                (
                    "{:.2%}".format(v)
                    if k in ["total_return", "max_drawdown"]
                    else "{:.2f}".format(v)
                    if k == "annualized_sharpe"
                    else "{:d}".format(v)
                ),
            )
            for k, v in metrics.items()
            if k
            in ["total_return", "max_drawdown", "annualized_sharpe", "number_of_trades"]
        )
        .replace("Total Return", "P&L(%)")
        .replace("Max Drawdown", "Max DD")
    )


def parse_mc_metrics_str(mc_metrics):
    return f"Annualized Sharpe: {mc_metrics['average_sharpe']:.2f} (SD: {mc_metrics['std_sharpe']:.2f}) | P&L(%): {mc_metrics['average_return']:.2%} (SD: {mc_metrics['std_return']:.2%}) | Max DD: {mc_metrics['average_max_drawdown']:.2%} (SD: {mc_metrics['std_max_drawdown']:.2%})"


def print_trade_log(trade):
    """
    Prints a single trade log to the console.

    Args:
        trade (pd.Series): A pandas Series representing a single trade.
                           Expected to have 'entry_date', 'exit_date',
                           'entry_price', 'exit_price', and 'return' columns.
    """
    entry_date = trade["entry_date"]
    exit_date = trade["exit_date"]
    entry_price = trade["entry_price"]
    exit_price = trade["exit_price"]
    side = trade["side"]
    trade_return = trade["return"]
    cum_return = trade["cum_return"]

    print("-" * 50)
    print(f"Side:         {side}")
    print(f"Entry:        {entry_date.date()} @ ${entry_price:,.2f}")
    print(f"Exit:         {exit_date.date()} @ ${exit_price:,.2f}")
    print(f"Trade Return: {trade_return:.2%}")
    print(f"Cum. Return:  {cum_return:.2%}")
    print("-" * 50)


def print_trade_logs(trades_df: pd.DataFrame):
    """
    Prints all trade logs from a DataFrame.

    Args:
        trades_df (pd.DataFrame): A DataFrame containing trade data.
    """
    if trades_df.empty:
        print("No trades to log.")
        return

    for _, trade in trades_df.iterrows():
        print_trade_log(trade)


def output_performance_summary(
    mc_backtester,
    benchmark_metrics: Dict,
):
    start_date = mc_backtester.test_df.index[0].date().strftime("%Y-%m-%d")
    end_date = mc_backtester.test_df.index[-1].date().strftime("%Y-%m-%d")
    print("=" * 100)
    print(
        f" Monte Carlo Metrics Over {mc_backtester.runs} Runs on Test Dataset (from {start_date} to {end_date})"
    )
    print("=" * 100)

    print(f"\n--- Benchmark (HODLing BTC from {start_date}) ---")
    print(
        parse_metrics_str(benchmark_metrics)
        .replace("Total Return", "P&L(%)")
        .replace("Max Drawdown", "Max DD(%)")
        .split(" | Number Of Trades:")[0]
    )

    print("\n--- Average HMM Strategy Path ---")
    print(
        parse_mc_metrics_str(mc_backtester.summary_statistics())
        .replace("Total Return", "P&L(%)")
        .replace("Max Drawdown", "Max DD(%)")
    )

    print("\n--- Outperformance Probabilities ---")
    print(
        f"- Beating HODLing:              {mc_backtester.probability_outperformance():.0%}"
    )
    print(
        f"- At least 2× HODLing returns:  {mc_backtester.probability_outperformance(2):.0%}"
    )
    print(
        f"- At least 3× HODLing returns:  {mc_backtester.probability_outperformance(3):.0%}"
    )
    print("=" * 100)
