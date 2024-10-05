from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import pywt
from scipy.fft import fft
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Step 1: Define classes for each step in the pipeline

class ImportData(BaseEstimator, TransformerMixin):
    def __init__(self, accel_path, reports_path):
        self.accel_path = accel_path
        self.reports_path = reports_path

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        raw_acceleration_data = pd.read_csv(self.accel_path)
        raw_selfreports_data = pd.read_csv(self.reports_path)
        df_accel = raw_acceleration_data.copy()
        df_reports = raw_selfreports_data.copy()
        print("Data imported successfully.")
        return df_reports, df_accel

class Preprocessing(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df_reports, df_accel = X
        df_reports = df_reports[df_reports.timeOfEngagement != 0].copy()
        df_reports = df_reports[df_reports.valence != "NONE"].copy()
        df_reports = df_reports[df_reports.arousal != "NONE"].copy()
        df_reports = df_reports[df_reports.context != "NONE"].copy()
        df_accel.rename(columns={'timestamp': 'timeOfNotification'}, inplace=True)
        df_accel["timeOfNotification"] = pd.to_datetime(df_accel["timeOfNotification"], unit="ms")
        df_reports["timeOfNotification"] = pd.to_datetime(df_reports["timeOfNotification"], unit="ms")
        print("Data preprocessed successfully.")
        return df_reports, df_accel

class ExtractAccelData(BaseEstimator, TransformerMixin):
    def __init__(self, time_window):
        self.time_window = time_window

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df_reports, df_accel = X
        df_reports['accel_data'] = df_reports.apply(lambda row: self._extract_accel_data(row, df_accel), axis=1)
        print("Accelerometer data extracted successfully.")
        return df_reports, df_accel

    def _extract_accel_data(self, row, accel_data):
        time_delta = pd.Timedelta(minutes=self.time_window)
        start_time = row['timeOfNotification'] - time_delta
        end_time = row['timeOfNotification'] + time_delta
        participant_id = row['participantId']
        mask = (
            (accel_data['participantId'] == participant_id) &
            (accel_data['timeOfNotification'] >= start_time) &
            (accel_data['timeOfNotification'] <= end_time)
        )
        return accel_data[mask]

class CreateCombinedDataFrame(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df_reports, df_accel = X
        combined_data = []

        for _, row in df_reports.iterrows():
            accel_data = row['accel_data']
            for _, accel_row in accel_data.iterrows():
                combined_data.append({
                    'participantId': row['participantId'],
                    'selfreport_time': row['timeOfNotification'],
                    'valence': row['valence'],
                    'arousal': row['arousal'],
                    'context': row['context'],
                    'accel_time': accel_row['timeOfNotification'],
                    'x': accel_row['x'],
                    'y': accel_row['y'],
                    'z': accel_row['z']
                })

        combined_df = pd.DataFrame(combined_data)
        print("Combined dataframe created successfully.")
        return combined_df

class CreateReportID(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['reportId'] = X.groupby(['participantId', 'selfreport_time']).ngroup() + 1
        col = X.pop("reportId")
        X.insert(0, col.name, col)
        print("Report ID created successfully.")
        return X

class ScaleXYZData(BaseEstimator, TransformerMixin):
    def __init__(self, scaler_type='standard'):
        self.scaler_type = scaler_type

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        columns_to_scale = ['x', 'y', 'z']
        if self.scaler_type == 'standard':
            scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid scaler_type. Expected 'standard' or 'minmax'.")
        scaled_columns = scaler.fit_transform(X[columns_to_scale])
        scaled_df = pd.DataFrame(scaled_columns, columns=columns_to_scale, index=X.index)
        X[columns_to_scale] = scaled_df
        print("Data scaled successfully.")
        return X

class PreprocessData(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.drop(['participantId', 'selfreport_time'], axis=1)
        X.rename(columns={"accel_time": "datetime"}, inplace=True)
        X['datetime'] = pd.DatetimeIndex(X["datetime"]).astype(np.int64) / 1000000  # Convert to Unix time in seconds
        print("Data preprocessed successfully.")
        return X

import numpy as np
import pandas as pd

class ExtractFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, window_length, window_step_size, data_frequency, selected_domains=None, include_magnitude=False):
        self.window_length = window_length
        self.window_step_size = window_step_size
        self.data_frequency = data_frequency
        self.selected_domains = selected_domains
        self.include_magnitude = include_magnitude

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features_list = []

        for report_id in X['reportId'].unique():
            temp = X[X['reportId'] == report_id]
            temp_ex = temp[['datetime', 'x', 'y', 'z']].copy()
            windows = self._window_data(temp_ex[['x', 'y', 'z']])
            for window in windows:
                features = self._extract_features_from_window(window)
                features['reportId'] = report_id
                features['arousal'] = temp['arousal'].iloc[0]
                features['valence'] = temp['valence'].iloc[0]
                features['context'] = temp['context'].iloc[0]
                features_list.append(pd.DataFrame([features]))  # Convert dictionary to DataFrame

        all_features = pd.concat(features_list, ignore_index=True) 

        print("All features extracted successfully.")
        return all_features

    def _window_data(self, data):
        window_samples = int(self.window_length * self.data_frequency)
        step_samples = int(self.window_step_size * self.data_frequency)
        windows = [data[i:i + window_samples] for i in range(0, len(data) - window_samples + 1, step_samples)]
        return np.array(windows)

    def _extract_features_from_window(self, window):
        all_features = {}

        if self.selected_domains is None or 'time_domain' in self.selected_domains:
            all_features.update(self._extract_time_domain_features(window))
        
        if self.selected_domains is None or 'spatial' in self.selected_domains:
            all_features.update(self._extract_spatial_features(window))
        
        # Add other domains (frequency, statistical, wavelet) as needed

        return all_features

    # Time Domain Features
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
        }

        # Signal Magnitude Area (SMA)
        features['sma'] = np.sum(np.abs(window[:, 0])) + np.sum(np.abs(window[:, 1])) + np.sum(np.abs(window[:, 2]))

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

        return features

    def _calculate_magnitude(self, window):
        return np.sqrt(window[:, 0]**2 + window[:, 1]**2 + window[:, 2]**2)

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
        
        return features

    def _calculate_magnitude(self, window):
        return np.sqrt(window[:, 0]**2 + window[:, 1]**2 + window[:, 2]**2)

# Step 2: Create the pipeline
pipeline = Pipeline([
    # Preprocessing Version vor Training
    ('import_datasets', ImportData(accel_path="C:/Users/duong/Desktop/BA/Analysis/dataset-main/AccelerometerMeasurements.csv",
                                   reports_path="C:/Users/duong/Desktop/BA/Analysis/dataset-main/SelfReports.csv")),
    ('preprocessing', Preprocessing()),
    ('extract_accel_data', ExtractAccelData(time_window=3)),
    ('create_combined_dataframe', CreateCombinedDataFrame()),
    # Live version
    ('label_with_report_id', CreateReportID()),
    ('scale_xyz_data', ScaleXYZData(scaler_type='standard')),
    ('preprocess_data', PreprocessData()),
    ('extract_features', ExtractFeatures(window_length=60, window_step_size=20, data_frequency=25, include_magnitude=True))
    # Classifier einfüegn
])

# Step 3: Run the pipeline
all_features = pipeline.fit_transform(None)
print("Pipeline executed successfully.")

# Step 4: Save the output to CSV
all_features.to_csv("Manual_60_20_Full.csv", encoding='utf-8', index=False)
print("Features saved successfully.")
