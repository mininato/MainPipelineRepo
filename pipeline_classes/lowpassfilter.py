import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import butter, filtfilt

class LowPassFilter(BaseEstimator, TransformerMixin):
    def __init__(self, cutoff_frequency, sampling_rate, order):
        """
        Initialize the LowPassFilter class.
        
        Parameters:
        - cutoff_frequency: The cutoff frequency for the low-pass filter (default: 5 Hz).
        - sampling_rate: The sampling rate of the accelerometer data (default: 25 Hz).
        - order: The order of the filter (default: 4).
        """
        self.cutoff_frequency = cutoff_frequency
        self.sampling_rate = sampling_rate
        self.order = order

    def _butter_lowpass_filter(self, data):
        """
        Apply a Butterworth low-pass filter to the data.
        
        Parameters:
        - data: A NumPy array containing the accelerometer data to be filtered.
        
        Returns:
        - A filtered NumPy array.
        """
        nyquist = 0.5 * self.sampling_rate
        normalized_cutoff = self.cutoff_frequency / nyquist
        b, a = butter(self.order, normalized_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data, axis=0)
        return filtered_data

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Apply the low-pass filter to the accelerometer data.
        
        Parameters:
        - X: A DataFrame with 'x', 'y', and 'z' columns representing the accelerometer data.
        
        Returns:
        - The DataFrame with filtered 'x', 'y', and 'z' columns.
        """
        if 'x' in X.columns and 'y' in X.columns and 'z' in X.columns:
            X[['x', 'y', 'z']] = self._butter_lowpass_filter(X[['x', 'y', 'z']].values)
            print("Low-pass filter applied successfully.")
        else:
            raise ValueError("The input DataFrame must contain 'x', 'y', and 'z' columns.")
        
        return X