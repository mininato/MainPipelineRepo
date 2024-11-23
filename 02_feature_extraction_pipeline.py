from sklearn.pipeline import Pipeline
from pipeline_classes import ImportData, LowPassFilter, ScaleXYZData, ExtractFeatures
from _config import config
import time

# This pipeline extracts features from the combined dataframe and exports it to a csv file
feature_extraction_pipeline = Pipeline([
    ('import_data', ImportData(use_accel=False, use_reports=False, use_combined=True, use_features=False)), # input path to combined data
    ('low_pass_filter', LowPassFilter(cutoff_frequency=config["cutoff_frequency"], sampling_rate=config["data_frequency"], order=config["order"])),
    ('scale_xyz_data', ScaleXYZData(scaler_type=config["scaler_type"])),
    ('extract_features', ExtractFeatures(window_length=config["window_length"],
                                         window_step_size=config["window_step_size"],
                                         data_frequency=config["data_frequency"],
                                         selected_domains=config["selected_domains"],
                                         include_magnitude=config["include_magnitude"],
                                         label_columns=config["label_columns"])),
])

# This will measure the time taken to run the pipeline
start_time = time.time()

# This will start the pipeline and return the feature dataframe
output_df = feature_extraction_pipeline.fit_transform(None)


end_time = time.time()
print(f"Time taken: {int((end_time - start_time) // 60)} minutes and {(end_time - start_time) % 60:.2f} seconds")
