import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from _config import config

class CreateCombinedDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, time_window):
        self.time_window = time_window

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        combined_data = []

        for _, row in X.iterrows():                                 #FIXME Very slow, need to optimize -> Future Work
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

        # Create groupid
        combined_df['groupid'] = combined_df.groupby(['participantId', 'selfreport_time']).ngroup() + 1    # Create unique groupid
        col = combined_df.pop("groupid")                                                                   # Move groupid to the first column
        combined_df.insert(0, col.name, col)

        # Create arousal_valence column
        combined_df['combined'] = combined_df['arousal'].astype(str) + "_" + combined_df['valence'].astype(str)

        # Drop rows with missing values
        combined_df.dropna(subset=['arousal', 'valence', 'groupid', 'combined'], inplace=True)

        # Export combined dataframe to CSV
        time_window_str = str(self.time_window)
        file_name = f"combined_data_timewindow_{time_window_str}min.csv"
        combined_df.to_csv(file_name, index=False)
        print(f"Combined dataframe exported successfully to {file_name}.")
        
        return combined_df