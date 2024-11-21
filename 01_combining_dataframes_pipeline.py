from sklearn.pipeline import Pipeline
from pipeline_classes import ImportData, CreateCombinedDataFrame
from _config import config
import time

combining_dataframes_pipeline = Pipeline([
    ('import_data', ImportData(use_accel=True, use_reports=True, use_combined=False, use_features=False)),  # input path to self-reports data),
    ('create_combined_dataframe', CreateCombinedDataFrame(time_window=config["time_window"], label_columns=config["label_columns"])),
])

start_time = time.time()
output_df = combining_dataframes_pipeline.fit_transform(None)
end_time = time.time()
print(f"Time taken: {int((end_time - start_time) // 60)} minutes and {(end_time - start_time) % 60:.2f} seconds")
