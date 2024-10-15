import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from _config import config

class ScaleXYZData(BaseEstimator, TransformerMixin):
    def __init__(self, scaler_type='standard'):
        self.scaler_type = scaler_type

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        columns_to_scale = ['x', 'y', 'z']                  
        if self.scaler_type == 'standard':                  # Scale the columns using StandardScaler or MinMaxScaler
            scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif self.scaler_type == 'none':
            return X  # Return the DataFrame without scaling
        else:
            raise ValueError("Invalid scaler_type. Expected 'standard' or 'minmax'.")   # Raise an error if scaler_type is invalid
        scaled_columns = scaler.fit_transform(X[columns_to_scale])
        scaled_df = pd.DataFrame(scaled_columns, columns=columns_to_scale, index=X.index)
        X[columns_to_scale] = scaled_df
        print("Data scaled successfully.")
        return X