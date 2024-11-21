import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from _config import config
    
class CreateCombinedDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, time_window, label_columns=None):
        self.time_window = time_window
        self.label_columns = label_columns #if label_columns else ["arousal", "valence"]  # Default to arousal and valence if not specified

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df_reports, df_accel = X

        print(f"PreprocesssingCombined initialized with label_columns: {self.label_columns}")
        # Ensure the chosen label columns exist in the dataset
        valid_conditions = (df_reports['timeOfEngagement'] != 0) 
        for label in self.label_columns:
            valid_conditions &= (df_reports[label] != "NONE")
        
        df_reports = df_reports[valid_conditions].copy()

        # Renaming and datetime conversion remains the same
        df_accel.rename(columns={'timestamp': 'timeOfNotification'}, inplace=True)
        df_accel["timeOfNotification"] = pd.to_datetime(df_accel["timeOfNotification"], unit="ms")
        df_reports["timeOfNotification"] = pd.to_datetime(df_reports["timeOfNotification"], unit="ms")

        print(f"ExtractAccelData initialized with time_window: {self.time_window}")
        df_reports, df_accel = X
        df_reports['accel_data'] = df_reports.apply(lambda row: self._extract_accel_data(row, df_accel), axis=1)
    
        print(f"Combining called with label_columns: {self.label_columns}")
        combined_data = []


        for _, row in df_reports.iterrows():
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
                    print(f"Processing label: {label}")
                    combined_row[label] = row[label]

                combined_data.append(combined_row)

        combined_df = pd.DataFrame(combined_data)

        # Create groupid column (unique identifier based on participantId and selfreport_time)
        combined_df['groupid'] = combined_df.groupby(['participantId', 'selfreport_time']).ngroup() + 1
        col = combined_df.pop("groupid")  # Move groupid to the first column
        combined_df.insert(0, col.name, col)

        # Export the combined dataframe to CSV
        time_window_str = str(self.time_window)
        label_columns_str = "_".join(self.label_columns)
        file_name = f"combined_data_timewindow_{time_window_str}min_labels_{label_columns_str}.csv"
        combined_df.to_csv(file_name, index=False)
        print(f"Combined dataframe exported successfully to {file_name}.")

        return combined_df
    
    def _extract_accel_data(self, row, accel_data):
        time_delta = pd.Timedelta(minutes=self.time_window)         
        start_time = pd.to_datetime(row['timeOfNotification']) - time_delta
        end_time = pd.to_datetime(row['timeOfNotification']) + time_delta
        participant_id = row['participantId']
        mask = (
            (accel_data['participantId'] == participant_id) &       # Filter out rows with different participantId
            (accel_data['timeOfNotification'] >= start_time) &      # Filter out rows with time outside the window
            (accel_data['timeOfNotification'] <= end_time)          # Filter out rows with time outside the window
        )
        return accel_data[mask] 