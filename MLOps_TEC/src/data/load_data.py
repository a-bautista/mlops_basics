import pandas as pd
from ucimlrepo import fetch_ucirepo

class LoadData:
    def __init__(self):
        pass

    def load_and_save_data(self, save_path):
        # Fetch dataset
        cdc_diabetes_health_indicators = fetch_ucirepo(id=891)
        X = cdc_diabetes_health_indicators.data.features
        y = cdc_diabetes_health_indicators.data.targets
        data = X.copy()
        data['Diabetes_binary'] = y
        data.to_csv(save_path, index=False)
        return data