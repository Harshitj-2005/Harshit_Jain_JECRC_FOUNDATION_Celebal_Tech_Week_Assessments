import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
from statsmodels.tsa.arima.model import ARIMA
import os
import pickle
import warnings

def train_arima_model(data_path='data/daily_city.csv', model_path='models/arima_daily.pkl'):
    # 1. Delete old pickle file if it exists
    if os.path.exists(model_path):
        try:
            os.remove(model_path)
            print(f"Removed old model file: {model_path}")
        except Exception as e:
            print(f"Could not remove old model file: {e}")

    # 2. Load data
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    # Explicitly set the frequency of the DatetimeIndex to Daily ('D')
    df.index.freq = 'D'
    
    series = df['energy_kwh']
    
    # 3. Fit ARIMA(1,1,1) model
    print("Fitting ARIMA(1, 1, 1) model on real daily energy consumption...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(series, order=(1,1,1))
        model_fit = model.fit()
    
    # 4. Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model_fit, f)
    
    print(f'New ARIMA model trained and saved successfully to {model_path}')
    return model_fit

if __name__ == "__main__":
    train_arima_model()
