import numpy as np
import pandas as pd

import pandas_ta as ta
from .config import (
    EMBARGO_PERIOD,
    ROLL_VOL,
    MOM_WINDOW,
    RSI_WINDOW,
    ADX_WINDOW,
    TRAIN_END_DATE,
)
from utils.logger import get_logger

logger = get_logger(__name__)

EXPECTED_FEATURES = [
    "ret",
    "vol21",
    "rsi",
    # "mom10",
    # "adx",
]


class FeatureEngineer:
    def __init__(
        self,
        roll_vol=ROLL_VOL,
        mom_window=MOM_WINDOW,
        rsi_window=RSI_WINDOW,
        adx_window=ADX_WINDOW,
    ):
        self.roll_vol = roll_vol
        self.mom_window = mom_window
        self.rsi_window = rsi_window
        self.adx_window = adx_window

    def build_features(
        self, df: pd.DataFrame, col: str = "Close"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Engineering Features...")
        df = df.copy()
        df["ret"] = np.log(df[col] / df[col].shift(1))
        df["vol21"] = df["ret"].rolling(self.roll_vol).std() * np.sqrt(365)
        df["rsi"] = ta.rsi(df[col], length=self.rsi_window)

        # df["mom10"] = df[col].pct_change(self.mom_window)
        # adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=self.adx_window)
        # df["adx"] = adx_df[f"ADX_{self.adx_window}"]

        df = df.dropna()

        features = df[EXPECTED_FEATURES].copy()
        logger.info("Features ready.")
        return df, features

    def split_data_into_train_test(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        split_date: str = TRAIN_END_DATE,
        embargo_period: int = EMBARGO_PERIOD,
    ):
        train_df = df.loc[:split_date].copy()
        test_df = df.loc[split_date:].copy()
        features_train = features.loc[:split_date].copy()
        features_test = features.loc[split_date:].copy()
        if embargo_period:
            train_df = train_df.iloc[:-embargo_period].copy()
            features_train = features_train.iloc[:-embargo_period].copy()

        logger.info(
            f"Train: {train_df.index.min()} → {train_df.index.max()} ({len(train_df)} days)"
        )
        if embargo_period:
            logger.info(
                f"Embargo: {train_df.index.max()} → {test_df.index.min()} ({embargo_period} days)"
            )
        logger.info(
            f"Test: {test_df.index.min()} → {test_df.index.max()} ({len(test_df)} days)"
        )
        return train_df, test_df, features_train, features_test
