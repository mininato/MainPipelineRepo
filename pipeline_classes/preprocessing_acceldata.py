import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from _config import config

class PreprocessingAccelData(BaseEstimator, TransformerMixin):
    def __init__(self, bin_size_minutes=3):
        self.bin_size_minutes = bin_size_minutes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df_accel = X

        # Calculate the bin size in terms of the number of samples
        bin_size_samples = self.bin_size_minutes * 60 * config["data_frequency"]

        # Create a new column for groupid
        df_accel['groupid'] = (df_accel.index // bin_size_samples) + 1
        print("Groupid created successfully.")

        return df_accel