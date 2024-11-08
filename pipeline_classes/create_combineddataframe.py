import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from _config import config
    
class CreateCombinedDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, time_window, label_columns=None):
        self.time_window = time_window
        self.label_columns = label_columns #if label_columns else ["arousal", "valence"]  # Default to arousal and valence if not specified
        print(f"Initialized with label_columns: {self.label_columns}")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(f"Transform called with label_columns: {self.label_columns}")
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
        file_name = f"combined_data_timewindow_{time_window_str}min.csv"
        combined_df.to_csv(file_name, index=False)
        print(f"Combined dataframe exported successfully to {file_name}.")

        return combined_df