import pandas as pd
import numpy as np
import time
from sklearn.base import BaseEstimator, TransformerMixin
from _config import config

class PreprocessingCombined(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df_reports, df_accel = X

        valid_conditions = (
            (df_reports['timeOfEngagement'] != 0) &     # Filter out rows with missing values
            (df_reports['valence'] != "NONE") &         # Filter out rows with 'NONE' values
            (df_reports['arousal'] != "NONE") &         # Filter out rows with 'NONE' values
            (df_reports['context'] != "NONE")           # Filter out rows with 'NONE' values
        )
        df_reports = df_reports[valid_conditions].copy()

        df_accel.rename(columns={'timestamp': 'timeOfNotification'}, inplace=True)
        df_accel["timeOfNotification"] = pd.to_datetime(df_accel["timeOfNotification"], unit="ms")
        df_reports["timeOfNotification"] = pd.to_datetime(df_reports["timeOfNotification"], unit="ms")
    
        return df_reports, df_accel