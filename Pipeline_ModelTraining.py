import pandas as pd
import numpy as np
import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.externals import joblib
from skopt.space import Real, Integer, Categorical
import pywt
from scipy.fft import fft
from scipy.signal import welch



# Configuration for default settings
config = {
    # Configuration for default settings

    # Configuration for data import
    "time_window": 3,  # Time window in minutes
    "scaler_type": "standard",  # 'standard' or 'minmax'

    # Configuration for feature extraction
    "window_length": 60,  # Window length in seconds
    "window_step_size": 20,  # Step size in seconds / 10%-50% of window_length
    "data_frequency": 25,  # Data frequency in Hz
    "selected_domains": None,  # Default: Every domain / 'time_domain', 'spatial', 'frequency', 'statistical', 'wavelet'
    "include_magnitude": True,  # Include magnitude-based features or not

    # Configuration for PCA
    "apply_pca": False,  # Apply PCA or not
    "pca_variance": 0.95,  # PCA variance threshold

    # Configuration for model training
    "classifier": "xgboost",  # Default classifier ('xgboost', 'svm', 'randomforest')
    "target": "combined",  # Target to train ('combined', 'arousal', 'valence')

    # Configuration for hyperparameter tuning
    "n_splits": 5,
    "n_iter": 30,
    "n_jobs": -1,
    "n_points": 1,

    # If users want to define custom param_space, they can specify it here
    "param_space": None,  # Set to None to use default inside the TrainModel class
}

# Pipeline Classes
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
        return accel_data[mask]                                     # Return the filtered rows

class CreateCombinedDataFrame(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        combined_data = []

        for _, row in X.iterrows():
            accel_data = row['accel_data']
            for _, accel_row in accel_data.iterrows():
                combined_data.append({
                    'participantId': row['participantId'],          # Add participantId to the combined data
                    'selfreport_time': row['timeOfNotification'],   # Add selfreport_time to the combined data
                    'valence': row['valence'],                      # Add valence to the combined data
                    'arousal': row['arousal'],                      # Add arousal to the combined data
                    'context': row['context'],                      # Add context to the combined data
                    'accel_time': accel_row['timeOfNotification'],  # Add accel_time to the combined data
                    'x': accel_row['x'],                            # Add x, y, z to the combined data
                    'y': accel_row['y'],
                    'z': accel_row['z']
                })

        combined_df = pd.DataFrame(combined_data)

        # Create reportId
        combined_df['reportId'] = combined_df.groupby(['participantId', 'selfreport_time']).ngroup() + 1    # Create unique reportId
        col = combined_df.pop("reportId")                                                                   # Move reportId to the first column
        combined_df.insert(0, col.name, col)

        # Create arousal_valence column
        combined_df['combined'] = combined_df['arousal'].astype(str) + "_" + combined_df['valence'].astype(str)

        # Drop rows with missing values
        combined_df.dropna(subset=['arousal', 'valence', 'reportId', 'combined'], inplace=True)

        # Export combined dataframe to CSV
        time_window_str = str(self.time_window)
        file_name = f"combined_data_timewindow_{time_window_str}min.csv"
        combined_df.to_csv(file_name, index=False)
        print(f"Combined dataframe exported successfully to {file_name}.")
        
        return combined_df

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
            temp = X[X['reportId'] == report_id]                            # takes all rows with the same reportId
            temp_ex = temp[['accel_time', 'x', 'y', 'z']].copy()            # takes only the columns needed #TIPS: acceltime can be removed 
            windows = self._window_data(temp_ex[['x', 'y', 'z']])           # creates windows of the data
            for window in windows:
                features = self._extract_features_from_window(window)       # extracts features from each window
                features['reportId'] = report_id                            # adds reportId to the features
                features['arousal'] = temp['arousal'].iloc[0]               # adds arousal and valence to the features
                features['valence'] = temp['valence'].iloc[0]               
                features['context'] = temp['context'].iloc[0]               # adds context to the features
                features['participantId'] = temp['participantId'].iloc[0]   # adds participantId to the features
                features["combined"] = temp["combined"].iloc[0]             # adds combined to the features
                features_list.append(pd.DataFrame([features]))              # Convert dictionary to DataFrame

        all_features = pd.concat(features_list, ignore_index=True)

        # Export features to CSV
        window_length_str = str(self.window_length)                 # Naming the file
        window_step_size_str = str(self.window_step_size)
        if self.selected_domains is None:                           # All features calculated if domains are not selected
            domain_str = "all_features"
        else:
            domain_str = "_".join(self.selected_domains)
        file_name = f"features_window_{window_length_str}_step_{window_step_size_str}_{domain_str}.csv"
        all_features.to_csv(file_name, index=False)

        print("All features extracted successfully.")
        return all_features

    def _window_data(self, data):                                                            # Function to create windows of the data
        window_samples = int(self.window_length * self.data_frequency)                       # Number of samples in each window 60sec * 25Hz = 1500 samples
        step_samples = int(self.window_step_size * self.data_frequency)                                             # Number of samples to move the window
        windows = [data[i:i + window_samples] for i in range(0, len(data) - window_samples + 1, step_samples)]      # Create windows
        return np.array(windows)

    def _extract_features_from_window(self, window):
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

    # Time Domain Features
    def _calculate_magnitude(self, window):
        return np.sqrt(window[:, 0]**2 + window[:, 1]**2 + window[:, 2]**2)

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
        print(f"Time domain features extracted successfully.")

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
            print(f"Additional time domain features for magnitude extracted successfully.")

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

        print(f"Spatial features extracted successfully.")
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

        print(f"Frequency domain features extracted successfully.")
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
        
        print(f"Statistical features extracted successfully.")
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
        
        print(f"Wavelet features extracted successfully.")
        return features

class PCAHandler(BaseEstimator, TransformerMixin):
    def __init__(self, apply_pca=False, variance=0.95):
        self.apply_pca = apply_pca
        self.variance = variance
        self.pca = None

    def fit(self, X, y=None):
        if self.apply_pca:
            self.pca = PCA(n_components=self.variance)
            self.pca.fit(X)
        return self

    def transform(self, X):
        if self.apply_pca and self.pca:
            X_transformed = self.pca.transform(X)
            return pd.DataFrame(X_transformed, index=X.index)
        
        return X

class TrainModel(BaseEstimator, TransformerMixin):
    def __init__(self, config, target="combined"):
        self.config = config
        self.target = config.get("target", "combined")  # Default to "combined"
        self.label_encoder = LabelEncoder()

    def fit(self, X, y):
        # Fit the label encoder on the target column
        print(f"Encoding the target labels for '{self.target}'...")
        self.label_encoder.fit(X[self.target])
        
        # Print the mapping between original labels and encoded labels
        original_labels = list(self.label_encoder.classes_)
        encoded_labels = list(range(len(original_labels)))
        label_mapping = dict(zip(encoded_labels, original_labels))
        print(f"Label encoding complete. Mapping: {label_mapping}")

        # Transform the target column and add it as 'target'
        X['target'] = self.label_encoder.transform(X[self.target])

        # Drop the original target column to avoid redundancy
        X = X.drop(columns=[self.target])

        # Choose classifier                     #TODO check paramspace of 
        classifier = self.config['classifier']
        if classifier == 'xgboost':
            model = XGBClassifier(objective='multi:softmax', random_state=42)
        elif classifier == 'svm':
            model = SVC(probability=True)
        elif classifier == 'randomforest':
            model = RandomForestClassifier(random_state=42)
        else:
            raise ValueError(f"Unsupported classifier type: {classifier}")
        print(f"Training the model using {self.config['classifier']}...")

        # Use user-defined param_space if provided, otherwise use default
        default_param_space = self.get_default_param_space(classifier)
        param_space = self.config.get("param_space", default_param_space)

        
        # Hyperparameter tuning using Bayesian optimization
        sgkf = StratifiedGroupKFold(n_splits=self.config['n_splits'])
        opt = BayesSearchCV(
            estimator=model,
            search_spaces=param_space,
            cv=sgkf,
            n_iter=self.config['n_iter'],
            n_jobs=self.config['n_jobs'],
            n_points=self.config['n_points'],
            verbose=1,
            scoring='accuracy'
        )

        print("Hyperparameter tuning in progress...")

        opt.fit(X.drop(columns=['target']), X['target'])
        self.best_model = opt.best_estimator_
        print(f"Best parameters found: {opt.best_params_}")

        # Save the best model
        model_name = f"{self.config['classifier']}_best_model_{self.config['model_name']}.pkl"
        joblib.dump(self.best_model, model_name)

        print("Model trained successfully.")
        return self

    def transform(self, X):
        return X
    
    # Configuration for hyperparameter tuning #TODO Check for paramspaces - which are the best for each model
    def get_default_param_space(classifier):
        # Returns the default hyperparameter space for a given classifier.
        if classifier == 'xgboost':
            return {
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'n_estimators': Integer(100, 1000),
                'max_depth': Integer(3, 10)
            }
        elif classifier == 'svm':
            return {
                'C': Real(0.1, 10, prior='log-uniform'),
                'kernel': Categorical(['linear', 'rbf'])
            }
        elif classifier == 'randomforest':
            return {
                'n_estimators': Integer(100, 1000),
                'max_depth': Integer(3, 10)
            }
        else:
            raise ValueError(f"Unsupported classifier type: {classifier}")

# Full training pipeline including every step
full_training_pipeline = Pipeline([
    ('import_data', ImportData(accel_path="path/to/AccelerometerData.csv", reports_path="path/to/SelfReports.csv")),
    ('preprocessing', Preprocessing()),
    ('extract_accel_data', ExtractAccelData(time_window=config["time_window"])),
    ('create_combined_dataframe', CreateCombinedDataFrame()),
    ('scale_xyz_data', ScaleXYZData(scaler_type=config["scaler_type"])),
    # ('preprocess_combined_dataframe', PreprocessCombinedDataframe()),
    ('extract_features', ExtractFeatures(window_length=config["window_length"], 
                                         window_step_size=config["window_step_size"], 
                                         data_frequency=config["data_frequency"], 
                                         selected_domains=config["selected_domains"], 
                                         include_magnitude=config["include_magnitude"])),
    ('pca_handler', PCAHandler(apply_pca=config["apply_pca"], variance=config["pca_variance"])),
    ('train_model', TrainModel(config=config)),
])

# Given Pipeline for combining dataframes
combining_dataframes_pipeline = Pipeline([
    ('import_data', ImportData()),
    ('preprocessing', Preprocessing()),
    ('extract_accel_data', ExtractAccelData(time_window=config["time_window"])),
    ('create_combined_dataframe', CreateCombinedDataFrame()),
])

# Given model training pipeline
training_model_pipeline = Pipeline([
    ('scale_xyz_data', ScaleXYZData(scaler_type=config["scaler_type"])),
    ('extract_features', ExtractFeatures(window_length=config["window_length"], 
                                         window_step_size=config["window_step_size"], 
                                         data_frequency=config["data_frequency"], 
                                         selected_domains=config["selected_domains"], 
                                         include_magnitude=config["include_magnitude"])),
    ('pca_handler', PCAHandler(apply_pca=config["apply_pca"], variance=config["pca_variance"])),
    ('train_model', TrainModel(config=config)),
])

accel_path = "C:/Users/duong/Documents/GitHub/MainPipelineRepo/AccelerometerMeasurements_backup.csv"
reports_path = "C:/Users/duong/Documents/GitHub/MainPipelineRepo/SelfReports_backup.csv"

# Run combining_dataframes_pipeline
start_time = time.time()
combining_dataframes_pipeline.fit_transform((None))
end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")


