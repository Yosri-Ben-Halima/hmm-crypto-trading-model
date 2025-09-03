import numpy as np
import pandas as pd

from utils.helpers import print_trade_logs
from .config import INITIAL_CAPITAL, COMMISSION, SLIPPAGE, MIN_HOLD_DAYS
from utils.logger import get_logger

logger = get_logger(__name__)


class Backtester:
    def __init__(
        self,
        initial_cap=INITIAL_CAPITAL,
        commission=COMMISSION,
        slippage=SLIPPAGE,
        min_hold_days=MIN_HOLD_DAYS,
    ):
        self.initial_cap = initial_cap
        self.commission = commission
        self.slippage = slippage
        self.min_hold_days = min_hold_days

    def backtest(self, df: pd.DataFrame, verbose=False) -> pd.DataFrame:
        df = df.copy()
        df["position"] = df["signal"].shift(1).fillna(0)  # act on yesterday's signal
        # Enforce min hold days (optional)
        if self.min_hold_days > 1:
            pos = df["position"].values.copy()
            last = 0
            hold = 0
            for i in range(len(pos)):
                if pos[i] == last:
                    hold += 1
                else:
                    last = pos[i]
                    hold = 1
                if hold < self.min_hold_days:
                    pos[i] = 0
            df["position"] = pos

        df["returns"] = np.exp(df["logret"]) - 1
        df["trade"] = df["position"].diff(-1).abs()
        df.loc[df.index[-1], "trade"] = df["signal"].iloc[-1]

        df["strategy_ret"] = df["position"] * df["returns"] - df["trade"] * (
            self.commission + self.slippage
        )

        position_diff = -df["position"].diff(-1)
        conditions = [
            position_diff > 0,
            position_diff < 0,
        ]
        choices = ["buy", "sell"]
        df["direction"] = np.select(conditions, choices, default="no action")
        df.dropna(inplace=True)

        df["hodl_position"] = 1
        df.loc[df.index[0], "hodl_position"] = 0
        df["hodl_ret"] = df["hodl_position"] * df["returns"]
        df.loc[df.index[0], "hodl_ret"] = -(self.commission + self.slippage)

        df["strategy_equity"] = self.initial_cap * (1 + df["strategy_ret"]).cumprod()

        df["hodl_equity"] = self.initial_cap * (1 + df["hodl_ret"]).cumprod()
        df["outperforming"] = df["strategy_equity"] >= df["hodl_equity"]
        if verbose:
            logger.info("HMM Strategy Backtesting & Trade Logs")
            self._log_trades(df)
        return df

    def _log_trades(self, df: pd.DataFrame):
        # keep only rows where a trade was executed
        trades = df[df["trade"] >= 1].copy()
        if trades.empty:
            return

        trades_list = []
        open_trade = None  # will hold the entry order
        prod = 1
        n_winning_trades = 0
        for i, row in trades.iterrows():
            direction = row["direction"].lower()
            price = row["Close"]

            if open_trade is None:
                # open a new trade
                open_trade = {
                    "entry_date": i,
                    "entry_price": price,
                    "side": "Long" if direction == "buy" else "Short",
                    "direction": direction,
                }
            else:
                # check if this closes the open trade (opposite direction)
                if (open_trade["direction"] == "buy" and direction == "sell") or (
                    open_trade["direction"] == "sell" and direction == "buy"
                ):
                    # Close trade
                    exit_date, exit_price = i, price
                    entry_price = open_trade["entry_price"]
                    side = open_trade["side"]

                    if side == "Long":
                        trade_return = (exit_price - entry_price) / entry_price
                    else:  # Short
                        trade_return = (entry_price - exit_price) / entry_price

                    # Apply costs
                    trade_return -= 2 * (self.commission + self.slippage)

                    prod *= 1 + trade_return
                    cum_return = prod - 1
                    n_winning_trades += 1 if trade_return >= 0 else 0

                    trades_list.append(
                        {
                            "entry_date": open_trade["entry_date"],
                            "exit_date": exit_date,
                            "side": side,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "return": trade_return,
                            "cum_return": cum_return,
                        }
                    )
                    open_trade = None
                else:
                    # Same direction twice in a row (invalid in strict sense).
                    # You can choose to overwrite entry or skip.
                    logger.warning(
                        f"Consecutive {direction.upper()} at {i} without closing prior trade. Skipping..."
                    )

        if not trades_list:
            return

        trades_df = pd.DataFrame(trades_list)
        logger.info(
            f"Winning Trades Percentage: {n_winning_trades / len(trades_list):.2%}"
        )
        print_trade_logs(trades_df)

    def metrics(self, df: pd.DataFrame, col: str = "strategy_equity") -> dict:
        ret_col = "strategy_ret" if col == "strategy_equity" else "returns"
        sr = df[ret_col].mean() / df[ret_col].std() * np.sqrt(365)
        total_return = df[col].iloc[-1] / df[col].iloc[0] - 1
        drawdown = 1 - df[col] / df[col].cummax()
        maxdd = -1 * drawdown.max()
        trades = int(df["trade"].sum()) if col == "strategy_equity" else 1
        return {
            "annualized_sharpe": float(sr),
            "total_return": float(total_return),
            "max_drawdown": float(maxdd),
            "number_of_trades": trades,
        }
