import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from _config import config

class ImportData(BaseEstimator, TransformerMixin):
    def __init__(self, accel_path=None, reports_path=None, combined_data_path=None, features_data_path=None):
        self.accel_path = accel_path
        self.reports_path = reports_path
        self.combined_data_path = combined_data_path
        self.features_data_path = features_data_path

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.features_data_path:  # If the path to features data is provided
            # Load the features dataframe
            features_df = pd.read_csv(self.features_data_path)
            print('Features dataframe imported successfully.')
            return features_df

        elif self.combined_data_path:  # If the path to combined data is provided
            # Load the combined dataframe
            combined_df = pd.read_csv(self.combined_data_path)
            print('Combined dataframe imported successfully.')
            return combined_df

        else:  # Otherwise, load the raw accelerometer and reports data
            if not self.accel_path:
                raise ValueError("accel_path needs to be provided if combined_data_path and features_data_path are not given.")
            
            # Load accelerometer data
            raw_acceleration_data = pd.read_csv(self.accel_path)
            df_accel = raw_acceleration_data.copy()

            if self.reports_path:
                # Load self-reports data if provided
                raw_selfreports_data = pd.read_csv(self.reports_path)
                df_reports = raw_selfreports_data.copy()
                print('Raw data (accelerometer and self-reports) imported successfully.')
                return df_reports, df_accel
            else:
                # Only accelerometer data provided
                print('Raw accelerometer data imported successfully.')
                return df_accel