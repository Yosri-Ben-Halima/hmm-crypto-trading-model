from typing import Literal
import pandas as pd
import requests
import numpy as np
from .config import TICKER  # , START, END
from utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    def __init__(
        self,
        ticker=TICKER,
    ):
        self.ticker = ticker
        self.limit = 1000
        self.api_url = (
            "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
        )

    def get_data(
        self,
        focus: Literal[
            "expansion",
            "reproduction",
            "limit",
        ] = "expansion",
    ) -> pd.DataFrame:
        """
        Fetches historical OHLCV data from Deribit using the requests library.
        """
        logger.info(f"Fetching data for {self.ticker} from Binance API...")
        if focus == "reproduction":
            df = pd.read_csv("raw_data.csv")
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        else:
            url = "https://api.binance.com/api/v3/klines"
            params = {"symbol": self.ticker, "interval": "1d", "limit": self.limit}
            r = requests.get(url, params=params)
            r.raise_for_status()
            data = r.json()

            # Convert to DataFrame
            last1000_df = pd.DataFrame(
                data,
                columns=[
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_asset_volume",
                    "number_of_trades",
                    "taker_buy_base_volume",
                    "taker_buy_quote_volume",
                    "ignore",
                ],
            )

            # Keep only needed columns and convert types
            last1000_df = last1000_df[
                ["open_time", "open", "high", "low", "close"]
            ].astype(float)
            last1000_df["date"] = pd.to_datetime(last1000_df["open_time"], unit="ms")
            last1000_df.set_index("date", inplace=True)
            last1000_df.drop(columns=["open_time"], inplace=True)
            last1000_df.columns = [col.capitalize() for col in last1000_df.columns]
            # Calculate log returns
            last1000_df["logret"] = np.log(
                last1000_df["Close"] / last1000_df["Close"].shift(1)
            )
            last1000_df = last1000_df.dropna()
            if focus == "limit":
                df = last1000_df
            elif focus == "expansion":
                stored_df = pd.read_csv("raw_data.csv")
                stored_df["date"] = pd.to_datetime(stored_df["date"])
                stored_df = stored_df.set_index("date")
                df = (
                    pd.concat(
                        [last1000_df, stored_df],  # ignore_index=False
                    )
                    .drop_duplicates(subset=["date"])
                    .set_index("date")
                )

        logger.info("Data fetched successfully.")
        return df
