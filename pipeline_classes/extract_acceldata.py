import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from _config import config

class ExtractAccelData(BaseEstimator, TransformerMixin):
    def __init__(self, time_window):
        self.time_window = time_window

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df_reports, df_accel = X
        df_reports['accel_data'] = df_reports.apply(lambda row: self._extract_accel_data(row, df_accel), axis=1)
        return df_reports

    def _extract_accel_data(self, row, accel_data):
        time_delta = pd.Timedelta(minutes=self.time_window)         
        start_time = row['timeOfNotification'] - time_delta
        end_time = row['timeOfNotification'] + time_delta
        participant_id = row['participantId']
        mask = (
            (accel_data['participantId'] == participant_id) &       # Filter out rows with different participantId
            (accel_data['timeOfNotification'] >= start_time) &      # Filter out rows with time outside the window
            (accel_data['timeOfNotification'] <= end_time)          # Filter out rows with time outside the window
        )
        return accel_data[mask] 