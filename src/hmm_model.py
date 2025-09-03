import pandas as pd
from hmmlearn.hmm import GaussianHMM

# from typing import Optional
from sklearn.preprocessing import StandardScaler

from utils.suppressor import suppress_stdout
from .config import N_STATES, SEED, INCLUDE_SHORTING
from utils.logger import get_logger
import warnings

# Ignore specific warning text
warnings.filterwarnings("ignore", message="Model is not converging")
logger = get_logger(__name__)


class HMMModel:
    def __init__(self, n_states=N_STATES, random_state=SEED):
        self.n_states = n_states
        self.random_state = random_state
        self.model = None
        self.scaler = None
        # self.converged = None

    def fit(
        self,
        features: pd.DataFrame,
        verbose=True,
    ):
        if verbose:
            logger.info("Fitting HMM...")
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(features.values)
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=500,
            random_state=self.random_state,
        )
        with suppress_stdout():
            self.model.fit(X)
            # self.converged = self.model.monitor_.converged
        hidden_states = self.model.predict(X)
        if verbose:
            logger.info("Fitting HMM Complete.")
        return hidden_states

    def predict(self, features: pd.DataFrame, verbose=True):
        if verbose:
            logger.info("Predicting hidden states...")
        if self.scaler is None or self.model is None:
            raise ValueError("Model must be fitted before prediction.")
        X = self.scaler.transform(features.values)
        hidden_states = self.model.predict(X)
        if verbose:
            logger.info("Prediction complete.")
        return hidden_states

    def regime_to_signal(
        self,
        df: pd.DataFrame,
        hidden_states,
        include_shorting=INCLUDE_SHORTING,
        verbose=True,
    ):
        if verbose:
            logger.info("Computing signals...")
        df = df.copy()
        df["state"] = hidden_states

        # Next-day returns
        df["next_ret"] = df["ret"].shift(-1)

        # Mean future return per state
        state_stats = df.groupby("state")["next_ret"].mean()

        # Assign signals: long if >0, short if <0
        df["signal"] = df["state"].map(
            lambda s: 1
            if state_stats[s] > 0
            else ((-1 if include_shorting else 0) if state_stats[s] < 0 else 0)
        )

        # Flatten MultiIndex if present
        if df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)
        if verbose:
            logger.info("Successfully computed signals.")

        return df.ffill(), state_stats
