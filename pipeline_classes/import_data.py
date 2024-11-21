import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from _config import config

class ImportData(BaseEstimator, TransformerMixin):
    def __init__(self, use_accel=True, use_reports=True, use_combined=False, use_features=False):
        # Dynamically assign file paths based on usage
        self.accel_path = config["accel_path"] if use_accel else None
        self.reports_path = config["reports_path"] if use_reports else None
        self.combined_data_path = config["combined_data_path"] if use_combined else None
        self.features_data_path = config["features_data_path"] if use_features else None

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
