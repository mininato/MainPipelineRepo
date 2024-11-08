import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.fftpack import fft
from scipy.signal import welch
import pywt
from _config import config

class ExtractFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, window_length, window_step_size, data_frequency, selected_domains=None, include_magnitude=False, label_columns=None):
        self.window_length = window_length
        self.window_step_size = window_step_size
        self.data_frequency = data_frequency
        self.selected_domains = selected_domains
        self.include_magnitude = include_magnitude
        self.label_columns = label_columns #if label_columns else ["arousal", "valence"]  # Default to arousal and valence if not specified

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features_list = []

        if 'groupid' in X.columns:  # Check for groupid column
            for groupid in X['groupid'].unique():  # Iterate over unique group IDs
                temp = X[X['groupid'] == groupid]  # Filter rows by group ID
                temp_ex = temp[['accel_time', 'x', 'y', 'z']].copy()  # Keep only the necessary columns (accel_time can be removed if unused)
                windows = self._window_data(temp_ex[['x', 'y', 'z']])  # Create windows of data
                
                for window in windows:
                    features = self._extract_features_from_window(window)  # Extract features from each window
                    features['groupid'] = groupid  # Add groupid to the features
                    
                    # Dynamically add emotion labels to the features
                    for label in self.label_columns:
                        features[label] = temp[label].iloc[0]
                    
                    features_list.append(pd.DataFrame([features]))  # Convert dictionary to DataFrame
                    
        else:  # In case there's no groupid, calculate features without it
            windows = self._window_data(X[['x', 'y', 'z']])
            for window in windows:
                features = self._extract_features_from_window(window)
                features_list.append(pd.DataFrame([features]))

        all_features = pd.concat(features_list, ignore_index=True)

        # Export features to CSV
        window_length_str = str(self.window_length)
        window_step_size_str = str(self.window_step_size)
        if self.selected_domains is None:  # All features calculated if domains are not selected
            domain_str = "all_features"
        else:
            domain_str = "_".join(self.selected_domains)
        file_name = f"features_window_{window_length_str}_step_{window_step_size_str}_{domain_str}.csv"
        all_features.to_csv(file_name, index=False)

        print("All features extracted successfully.")
        return all_features

    # Time Domain Features
    def _calculate_magnitude(self, window):
        return np.sqrt(window[:, 0]**2 + window[:, 1]**2 + window[:, 2]**2)
    
    def _window_data(self, data):                                                            # Function to create windows of the data
        window_samples = int(self.window_length * self.data_frequency)                       # Number of samples in each window 60sec * 25Hz = 1500 samples
        step_samples = int(self.window_step_size * self.data_frequency)                                             # Number of samples to move the window
        windows = [data[i:i + window_samples] for i in range(0, len(data) - window_samples + 1, step_samples)]      # Create windows
        return np.array(windows)

    def _extract_features_from_window(self, window):                        #DONE Mehrere domains gleichzeitig berechnen
        all_features = {}

        if self.selected_domains is None or 'time_domain' in self.selected_domains:
            all_features.update(self._extract_time_domain_features(window))
        
        if self.selected_domains is None or 'spatial' in self.selected_domains:
            all_features.update(self._extract_spatial_features(window))
        
        if self.selected_domains is None or 'frequency' in self.selected_domains:
            all_features.update(self._extract_frequency_domain_features(window))

        if self.selected_domains is None or 'statistical' in self.selected_domains:
            all_features.update(self._extract_statistical_features(window))

        if self.selected_domains is None or 'wavelet' in self.selected_domains:
            all_features.update(self._extract_wavelet_features(window))

        return all_features

    def _extract_time_domain_features(self, window):
        features = {
            'mean_x': np.mean(window[:, 0]),
            'mean_y': np.mean(window[:, 1]),
            'mean_z': np.mean(window[:, 2]),
            'std_x': np.std(window[:, 0]),
            'std_y': np.std(window[:, 1]),
            'std_z': np.std(window[:, 2]),
            'variance_x': np.var(window[:, 0]),
            'variance_y': np.var(window[:, 1]),
            'variance_z': np.var(window[:, 2]),
            'rms_x': np.sqrt(np.mean(window[:, 0]**2)),
            'rms_y': np.sqrt(np.mean(window[:, 1]**2)),
            'rms_z': np.sqrt(np.mean(window[:, 2]**2)),
            'max_x': np.max(window[:, 0]),
            'max_y': np.max(window[:, 1]),
            'max_z': np.max(window[:, 2]),
            'min_x': np.min(window[:, 0]),
            'min_y': np.min(window[:, 1]),
            'min_z': np.min(window[:, 2]),
            'peak_to_peak_x': np.ptp(window[:, 0]),
            'peak_to_peak_y': np.ptp(window[:, 1]),
            'peak_to_peak_z': np.ptp(window[:, 2]),
            'skewness_x': pd.Series(window[:, 0]).skew(),
            'skewness_y': pd.Series(window[:, 1]).skew(),
            'skewness_z': pd.Series(window[:, 2]).skew(),
            'kurtosis_x': pd.Series(window[:, 0]).kurt(),
            'kurtosis_y': pd.Series(window[:, 1]).kurt(),
            'kurtosis_z': pd.Series(window[:, 2]).kurt(),
            'zero_crossing_rate_x': np.sum(np.diff(np.sign(window[:, 0])) != 0),
            'zero_crossing_rate_y': np.sum(np.diff(np.sign(window[:, 1])) != 0),
            'zero_crossing_rate_z': np.sum(np.diff(np.sign(window[:, 2])) != 0),
            'sma' : np.sum(np.abs(window[:, 0])) + np.sum(np.abs(window[:, 1])) + np.sum(np.abs(window[:, 2])), #Signal Magnitude Area
        }
        # print(f"Time domain features extracted successfully.")

        # Additional features for Magnitude (xyz in one vector)
        if self.include_magnitude:
            magnitude = self._calculate_magnitude(window)
            features['mean_magnitude'] = np.mean(magnitude)
            features['std_magnitude'] = np.std(magnitude)
            features['variance_magnitude'] = np.var(magnitude)
            features['rms_magnitude'] = np.sqrt(np.mean(magnitude**2))
            features['max_magnitude'] = np.max(magnitude)
            features['min_magnitude'] = np.min(magnitude)
            features['peak_to_peak_magnitude'] = np.ptp(magnitude)
            features['skewness_magnitude'] = pd.Series(magnitude).skew()
            features['kurtosis_magnitude'] = pd.Series(magnitude).kurt()
            features['zero_crossing_rate_magnitude'] = np.sum(np.diff(np.sign(magnitude)) != 0)
            # print(f"Additional time domain features for magnitude extracted successfully.")

        return features

    # Spatial Features
    def _extract_spatial_features(self, window):
        features = {}

        # Euclidean Norm (Magnitude)
        magnitude = self._calculate_magnitude(window)
        features['euclidean_norm'] = np.mean(magnitude)  # or np.linalg.norm for each window

        # Tilt Angles (Pitch and Roll)
        pitch = np.arctan2(window[:, 1], np.sqrt(window[:, 0]**2 + window[:, 2]**2)) * (180 / np.pi)
        roll = np.arctan2(window[:, 0], np.sqrt(window[:, 1]**2 + window[:, 2]**2)) * (180 / np.pi)
        features['mean_pitch'] = np.mean(pitch)
        features['mean_roll'] = np.mean(roll)

        # Correlation between Axes
        features['correlation_xy'] = np.corrcoef(window[:, 0], window[:, 1])[0, 1]
        features['correlation_xz'] = np.corrcoef(window[:, 0], window[:, 2])[0, 1]
        features['correlation_yz'] = np.corrcoef(window[:, 1], window[:, 2])[0, 1]

        # print(f"Spatial features extracted successfully.")
        return features



    # Frequency Domain Features
    def _extract_frequency_domain_features(self, window):
        n = len(window)
        freq_values = np.fft.fftfreq(n, d=1/self.data_frequency)[:n // 2]
        fft_values = fft(window, axis=0)
        fft_magnitude = np.abs(fft_values)[:n // 2]

        features = {}

        # Spectral Entropy
        def spectral_entropy(signal):
            psd = np.square(signal)
            psd_norm = psd / np.sum(psd)
            return -np.sum(psd_norm * np.log(psd_norm + 1e-10))

        for i, axis in enumerate(['x', 'y', 'z']):
            # Dominant Frequency
            dominant_frequency = freq_values[np.argmax(fft_magnitude[:, i])]
            features[f'dominant_frequency_{axis}'] = dominant_frequency

            # Spectral Entropy
            entropy = spectral_entropy(fft_magnitude[:, i])
            features[f'spectral_entropy_{axis}'] = entropy

            # Power Spectral Density (PSD) and Energy
            f, psd_values = welch(window[:, i], fs=self.data_frequency, nperseg=n)
            features[f'psd_mean_{axis}'] = np.mean(psd_values)
            features[f'energy_{axis}'] = np.sum(psd_values**2)

            # Bandwidth (frequency range containing significant portion of the energy)
            cumulative_energy = np.cumsum(psd_values)
            total_energy = cumulative_energy[-1]
            low_cutoff_idx = np.argmax(cumulative_energy > 0.1 * total_energy)
            high_cutoff_idx = np.argmax(cumulative_energy > 0.9 * total_energy)
            bandwidth = f[high_cutoff_idx] - f[low_cutoff_idx]
            features[f'bandwidth_{axis}'] = bandwidth

            # Spectral Centroid (Center of mass of the spectrum)
            spectral_centroid = np.sum(f * psd_values) / np.sum(psd_values)
            features[f'spectral_centroid_{axis}'] = spectral_centroid

        if self.include_magnitude:
            # Magnitude-based Frequency Domain Features
            magnitude = self._calculate_magnitude(window)
            fft_magnitude_mag = np.abs(fft(magnitude))[:n // 2]

            # Dominant Frequency for Magnitude
            features['dominant_frequency_magnitude'] = freq_values[np.argmax(fft_magnitude_mag)]

            # Spectral Entropy for Magnitude
            features['spectral_entropy_magnitude'] = spectral_entropy(fft_magnitude_mag)

            # Power Spectral Density and Energy for Magnitude
            f, psd_values_mag = welch(magnitude, fs=self.data_frequency, nperseg=n)
            features['psd_mean_magnitude'] = np.mean(psd_values_mag)
            features['energy_magnitude'] = np.sum(psd_values_mag**2)

            # Bandwidth for Magnitude
            cumulative_energy_mag = np.cumsum(psd_values_mag)
            total_energy_mag = cumulative_energy_mag[-1]
            low_cutoff_idx_mag = np.argmax(cumulative_energy_mag > 0.1 * total_energy_mag)
            high_cutoff_idx_mag = np.argmax(cumulative_energy_mag > 0.9 * total_energy_mag)
            bandwidth_mag = f[high_cutoff_idx_mag] - f[low_cutoff_idx_mag]
            features['bandwidth_magnitude'] = bandwidth_mag

            # Spectral Centroid for Magnitude
            features['spectral_centroid_magnitude'] = np.sum(f * psd_values_mag) / np.sum(psd_values_mag)

        # print(f"Frequency domain features extracted successfully.")
        return features


    def _extract_statistical_features(self, window):
        features = {
            '25th_percentile_x': np.percentile(window[:, 0], 25),
            '25th_percentile_y': np.percentile(window[:, 1], 25),
            '25th_percentile_z': np.percentile(window[:, 2], 25),
            '75th_percentile_x': np.percentile(window[:, 0], 75),
            '75th_percentile_y': np.percentile(window[:, 1], 75),
            '75th_percentile_z': np.percentile(window[:, 2], 75),
        }
        
        if self.include_magnitude:
            magnitude = self._calculate_magnitude(window)
            features['25th_percentile_magnitude'] = np.percentile(magnitude, 25)
            features['75th_percentile_magnitude'] = np.percentile(magnitude, 75)
        
        # print(f"Statistical features extracted successfully.")
        return features

    def _extract_wavelet_features(self, window, wavelet='db1'):
        coeffs = pywt.wavedec(window, wavelet, axis=0, level=3)
        features = {
            'wavelet_energy_approx_x': np.sum(coeffs[0][:, 0]**2),
            'wavelet_energy_approx_y': np.sum(coeffs[0][:, 1]**2),
            'wavelet_energy_approx_z': np.sum(coeffs[0][:, 2]**2),
        }
        
        if self.include_magnitude:
            magnitude = self._calculate_magnitude(window)
            coeffs_magnitude = pywt.wavedec(magnitude, wavelet, level=3)
            features['wavelet_energy_approx_magnitude'] = np.sum(coeffs_magnitude[0]**2)
        
        # print(f"Wavelet features extracted successfully.")
        return features