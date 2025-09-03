# import joblib
# from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.hmm_model import HMMModel


def train(raw_data):
    """
    Trains the HMM model and saves it to disk.
    """
    # 1. Load data
    # data_loader = DataLoader()
    # raw_data = data_loader.get_data()

    # 2. Feature Engineering
    feature_engineer = FeatureEngineer()
    df, features = feature_engineer.build_features(raw_data)

    # 3. Fit HMM
    hmm_model = HMMModel()
    hmm_model.fit(features)

    # 4. Save the model and scaler
    # joblib.dump(hmm_model.model, 'hmm_model.pkl')
    # joblib.dump(hmm_model.scaler, 'scaler.pkl')

    # print("Model trained and saved successfully.")
    return hmm_model.model, hmm_model.scaler


# if __name__ == "__main__":
#     train()
