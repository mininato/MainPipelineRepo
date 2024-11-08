import pandas as pd
import numpy as np
import time
from sklearn.base import BaseEstimator, TransformerMixin
from _config import config
    
class PreprocessingCombined(BaseEstimator, TransformerMixin):
    def __init__(self, label_columns):
        self.label_columns =label_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df_reports, df_accel = X

        # Ensure the chosen label columns exist in the dataset
        valid_conditions = (df_reports['timeOfEngagement'] != 0) 
        for label in self.label_columns:
            valid_conditions &= (df_reports[label] != "NONE")
        
        df_reports = df_reports[valid_conditions].copy()

        # Renaming and datetime conversion remains the same
        df_accel.rename(columns={'timestamp': 'timeOfNotification'}, inplace=True)
        df_accel["timeOfNotification"] = pd.to_datetime(df_accel["timeOfNotification"], unit="ms")
        df_reports["timeOfNotification"] = pd.to_datetime(df_reports["timeOfNotification"], unit="ms")

        return df_reports, df_accel