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
import joblib
from skopt.space import Real, Integer, Categorical
import pywt
from scipy.fft import fft
from scipy.signal import welch
from sklearn.metrics import classification_report, accuracy_score
import json



# Configuration for default settings
config = {
    # Configuration for default settings
    "label_columns": ["valence", "arousal"],  # Columns for target labels
    "target_label": "arousal",  # Target label for training

    # Configuration for data import
    # "accel_path": "C:/Users/duong/Documents/GitHub/MainPipelineRepo/AccelerometerMeasurements_backup.csv",  # Path to the accelerometer data
    # "reports_path": "C:/Users/duong/Documents/GitHub/MainPipelineRepo/SelfReports_backup.csv",  # Path to the self-reports data
    "time_window": 2,  # Time window in minutes
    "scaler_type": "standard",  # 'standard' or 'minmax'
    "bin_size_minutes": 3,  # Bin size in minutes

    # Configuration for feature extraction
    "window_length": 120,  # Window length in seconds / 60
    "window_step_size": 120,  # Step size in seconds / 10%-50% of window_length / 20
    "data_frequency": 25,  # Data frequency in Hz
    "selected_domains": None,  # Default: Every domain / 'time_domain', 'spatial', 'frequency', 'statistical', 'wavelet' / multiple domains: ["time_domain", "frequency"] / order is not important
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

class PreprocessingCombined(BaseEstimator, TransformerMixin):
    def __init__(self, label_columns):
        self.label_columns = label_columns

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
    def __init__(self, time_window, label_columns=None):
        self.time_window = time_window
        self.label_columns = label_columns #if label_columns else ["arousal", "valence"]  # Default to arousal and valence if not specified

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        combined_data = []

        for _, row in X.iterrows():
            accel_data = row['accel_data']
            for _, accel_row in accel_data.iterrows():
                combined_row = {
                    'participantId': row['participantId'],  # Participant ID
                    'selfreport_time': row['timeOfNotification'],  # Self-report time
                    'accel_time': accel_row['timeOfNotification'],  # Accelerometer data time
                    'x': accel_row['x'],  # x-axis accelerometer data
                    'y': accel_row['y'],  # y-axis accelerometer data
                    'z': accel_row['z']   # z-axis accelerometer data
                }

                # Dynamically add emotion labels to the combined row
                for label in self.label_columns:
                    combined_row[label] = row[label]

                combined_data.append(combined_row)

        combined_df = pd.DataFrame(combined_data)

        # Create groupid column (unique identifier based on participantId and selfreport_time)
        combined_df['groupid'] = combined_df.groupby(['participantId', 'selfreport_time']).ngroup() + 1
        col = combined_df.pop("groupid")  # Move groupid to the first column
        combined_df.insert(0, col.name, col)

        # Export the combined dataframe to CSV
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
    def __init__(self, config):
        self.config = config
        self.target = config.get("target_label", None)  # User-defined target label in config
        self.label_encoder = LabelEncoder()
        self.selected_domains = self.config.get("selected_domains", "All domains")  # Default to all domains if None

        if not self.target:
            raise ValueError("No target label specified in the config. Please set 'target_label'.")

    def get_default_param_space(self, classifier):
        """ Returns the default hyperparameter space for a given classifier. """
        if classifier == 'xgboost':
            return {
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'n_estimators': Integer(100, 1000),
                'max_depth': Integer(3, 10),
                'min_child_weight': (1, 10),
                'subsample': (0.5, 1.0),
                'colsample_bytree': (0.5, 1.0),
                'gamma': (0, 10),
                'reg_alpha': (0, 10),
                'reg_lambda': (0, 10),
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

    def fit(self, X, y=None):
        # Ensure the target column exists in the dataset
        if self.target not in X.columns:
            raise ValueError(f"Target label '{self.target}' not found in the dataset.")
        
        # Fit the label encoder on the target column
        print(f"Encoding the target labels for '{self.target}'...")
        self.label_encoder.fit(X[self.target])

        # Print the mapping between original labels and encoded labels
        original_labels = list(self.label_encoder.classes_)
        encoded_labels = list(range(len(original_labels)))
        label_mapping = dict(zip(encoded_labels, original_labels))
        print(f"Label encoding complete. Mapping: {label_mapping}")

        # Transform the target column and add it as 'encoded_target'
        X['encoded_target'] = self.label_encoder.transform(X[self.target])

        # Value counts for the encoded target        
        value_counts = X['encoded_target'].value_counts().to_dict()
        print(f"Value counts for encoded target: {value_counts}")
        
        # Pop unnecessary columns (groupid, emotion labels not being used, etc.)
        groups = X.pop('groupid')

        # Pop the encoded target as Y
        y = X.pop('encoded_target')

        # Store the feature names for later use
        feature_names = X.columns.tolist()

        # Choose classifier
        classifier = self.config['classifier']
        if classifier == 'xgboost':
            model = XGBClassifier(objective='multi:softmax', random_state=42)
        elif classifier == 'svm':
            model = SVC(probability=True)
        elif classifier == 'randomforest':
            model = RandomForestClassifier(random_state=42)
        else:
            raise ValueError(f"Unsupported classifier type: {classifier}")

        print(f"Training the model using {classifier}...")

        # Use user-defined param_space if provided, otherwise use default
        print(f"Classifier: {classifier}")
        default_param_space = self.get_default_param_space(classifier)
        param_space = self.config.get("param_space") or default_param_space     

        # Hyperparameter tuning using Bayesian optimization
        sgkf = StratifiedGroupKFold(n_splits=self.config['n_splits'])
        print(f"Parameter space being used: {param_space}")
        if param_space is None:
            raise ValueError("Parameter space cannot be None. Please check the classifier configuration.")

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

        # Fit the model using the encoded target
        opt.fit(X, y, groups=groups)
        self.best_model = opt.best_estimator_
        print(f"Best parameters found: {opt.best_params_}")

        # Print classification metrics
        y_pred = self.best_model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, target_names=self.label_encoder.classes_, output_dict=True)
        
        # Save classification report
        classification_report_json = report  
        with open(f'classification_report_{self.target}.json', 'w') as f:
            json.dump(classification_report_json, f, indent=4)

        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{report}")

        # Save the best model with the target label in the file name
        model_name = f"{classifier}_best_model_{self.target}.pkl"
        joblib.dump(self.best_model, model_name)
        print("Model saved successfully.")

        # Save model metadata
        model_metadata = {
            "best_params": opt.best_params_,
            "accuracy": accuracy,
            "classification_report": classification_report_json,
            "label_mapping": label_mapping,
            "model_name": model_name,
            "value_counts": value_counts,
            "selected_domains": self.selected_domains,  
            "include_magnitude": self.config.get("include_magnitude", True)      
        }

        if hasattr(self.best_model, "feature_importances_"):
            feature_importances = self.best_model.feature_importances_
            # Convert feature importances to native Python floats
            feature_importance_dict = {feature: float(importance) for feature, importance in zip(feature_names, feature_importances)}
            model_metadata["feature_importances"] = feature_importance_dict
            print("Feature Importances:")
            for feature, importance in feature_importance_dict.items():
                print(f"{feature}: {importance:.4f}")

        # Save metadata with the target name in the file name
        metadata_file = f"{classifier}_model_metadata_{self.target}.json"
        with open(metadata_file, "w") as f:
            json.dump(model_metadata, f, indent=4)
            print(f"Model metadata saved to {metadata_file}.")

        return self

    def transform(self, X):
        return X  # Placeholder for transform step (not needed for training)

class ClassifyMovementData(BaseEstimator, TransformerMixin):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.model is None:
            self.model = joblib.load(self.model_path)  # Load the pre-trained model
            print(f"Model loaded from {self.model_path}")

        # Assuming `X` is a DataFrame of pre-extracted features.
        predictions = self.model.predict(X)

        # Adding predictions to the DataFrame
        X['predicted_emotion'] = predictions

        print("Data classified successfully.")
        
        # Export the labeled DataFrame to CSV
        window_length_str = str(config["window_length"])
        output_file = f"classified_movement_data_window_{window_length_str}.csv"
        X.to_csv(output_file, index=False)
        print(f"Classified movement data exported successfully to {output_file}.")

        return X

#TODO Eigene User-individualisierte Labels, Falls user nicht das gleiche Valence_Arousal Model benutzt
#TODO Test ob individualisierte Labels funktionieren
#TODO Ordner erstellen für Klassen, Pro Klasse eine Datei, damit es wie ein Package fungiert, __init__.py erstellen, Oder pro Pipeline Modul eine Datei erstellen = Clean Code

# class UseModel

# Full training pipeline including every step
# full_training_pipeline = Pipeline([
#     ('import_data', ImportData(accel_path="path/to/AccelerometerData.csv", reports_path="path/to/SelfReports.csv")),
#     ('preprocessing', Preprocessing()),
#     ('extract_accel_data', ExtractAccelData(time_window=config["time_window"])),
#     ('create_combined_dataframe', CreateCombinedDataFrame()),
#     ('scale_xyz_data', ScaleXYZData(scaler_type=config["scaler_type"])),
#     # ('preprocess_combined_dataframe', PreprocessCombinedDataframe()),
#     ('extract_features', ExtractFeatures(window_length=config["window_length"], 
#                                          window_step_size=config["window_step_size"], 
#                                          data_frequency=config["data_frequency"], 
#                                          selected_domains=config["selected_domains"], 
#                                          include_magnitude=config["include_magnitude"])),
#     ('pca_handler', PCAHandler(apply_pca=config["apply_pca"], variance=config["pca_variance"])),
#     ('train_model', TrainModel(config=config)),
# ])

# Given Pipeline for combining dataframes
# First pipeline part (takes raw dataframes as input)
# combining_dataframes_pipeline = Pipeline([
#     ('import_data', ImportData(accel_path="C:/Users/duong/Documents/GitHub/MainPipelineRepo/AccelerometerMeasurements_backup.csv", # input path to accelerometer data
#                                reports_path="C:/Users/duong/Documents/GitHub/MainPipelineRepo/SelfReports_backup.csv")),            # input path to self-reports data),
#     ('preprocessing', PreprocessingCombined()),
#     ('extract_accel_data', ExtractAccelData(time_window=config["time_window"])),
#     ('create_combined_dataframe', CreateCombinedDataFrame(time_window=config["time_window"])),
# ])

# Feature extraction pipeline part (takes combined dataframe as input)
# feature_extraction_pipeline = Pipeline([
#     ('import_data', ImportData(combined_data_path="C:/Users/duong/Documents/GitHub/MainPipelineRepo/combined_data_timewindow_3min.csv")), # input path to combined data
#     ('scale_xyz_data', ScaleXYZData(scaler_type=config["scaler_type"])),
#     ('extract_features', ExtractFeatures(window_length=config["window_length"],
#                                          window_step_size=config["window_step_size"],
#                                          data_frequency=config["data_frequency"],
#                                          selected_domains=config["selected_domains"],
#                                          include_magnitude=config["include_magnitude"])),
# ])

# # Training model pipeline part (takes features dataframe as input)
# training_model_pipeline = Pipeline([
#     ('import_data', ImportData(features_data_path="C:/Users/duong/Documents/GitHub/MainPipelineRepo/features_window_60_step_20_all_features.csv")),
#     ('pca_handler', PCAHandler(apply_pca=config["apply_pca"], variance=config["pca_variance"])),
#     ('train_model', TrainModel(config=config)),
# ])

# accel_path = "C:/Users/duong/Documents/GitHub/MainPipelineRepo/AccelerometerMeasurements_backup.csv"
# reports_path = "C:/Users/duong/Documents/GitHub/MainPipelineRepo/SelfReports_backup.csv"
# combined_data_path = "C:/Users/duong/Documents/GitHub/MainPipelineRepo/combined_data_timewindow_3min.csv"

# Test user Pipeline
user_pipeline = Pipeline([
    ('import_data', ImportData(accel_path="C:/Users/duong/Documents/GitHub/MainPipelineRepo/single_participant_positive_high.csv")), # input path to accelerometer data)
    # ('preprocessing', PreprocessingAccelData(bin_size_minutes=config["bin_size_minutes"])),
    ('scale_xyz_data', ScaleXYZData(scaler_type=config["scaler_type"])),
    ('extract_features', ExtractFeatures(window_length=config['window_length'], window_step_size=config["window_step_size"], data_frequency=config["data_frequency"],
                                          selected_domains=config['selected_domains'], include_magnitude=config['include_magnitude'])),
    ('classify_movement_data', ClassifyMovementData(model_path="C:/Users/duong/Documents/GitHub/MainPipelineRepo/xgboost_best_model_combined.pkl")),
])

# Run training_model_pipeline
start_time = time.time()
output_df = user_pipeline.fit_transform(None)
end_time = time.time()
print(f"Time taken: {int((end_time - start_time) // 60)} minutes and {(end_time - start_time) % 60:.2f} seconds")

output_file = "user_pipeline_output.csv"
output_df.to_csv(output_file, index=False)
print(f"User pipeline output exported successfully to {output_file}.")


