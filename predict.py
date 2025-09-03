# import joblib
# import pandas as pd
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.hmm_model import HMMModel
from train import train


def predict():
    """
    Loads the trained HMM model, gets the latest data, and predicts the signal.
    """
    # 1. Load the trained model and scaler

    # 2. Load data (in a real scenario, you would fetch live data here)
    data_loader = DataLoader()
    raw_data = data_loader.get_data()

    # 3. Feature Engineering for the latest data point
    feature_engineer = FeatureEngineer()
    df, features = feature_engineer.build_features(raw_data)
    latest_features = features.tail(1)
    try:
        model, scaler = train(raw_data)
    except FileNotFoundError:
        print("Model not found. Please run train.py first.")
        return

    # 4. Predict the state for the latest data point
    scaled_features = scaler.transform(latest_features.values)
    hidden_state = model.predict(scaled_features)[0]

    # 5. Generate the signal
    hmm_model = HMMModel()
    hmm_model.model = model
    _, state_stats = hmm_model.regime_to_signal(
        df, model.predict(scaler.transform(features.values))
    )

    signal = 1 if state_stats[hidden_state] > 0 else 0

    # print(f"Predicted State: {hidden_state}")
    # print(f"State Stats (Mean Future Return):\n{state_stats}")
    # print("-" * 30)
    if signal == 1:
        print("Signal for next day: BUY/HOLD")
    else:
        print("Signal for next day: SELL/STAY NEUTRAL")


if __name__ == "__main__":
    predict()
