from sklearn.pipeline import Pipeline
from pipeline_classes import ImportData, LowPassFilter, ScaleXYZData, ExtractFeatures, ClassifyMovementData
from _config import config
import time

# This is the pipeline that will be used to analyze data which hasnt been classified yet and export the classified dataframe as a csv file
analyzing_data_pipeline = Pipeline([
    ('import_data', ImportData(use_accel=True, use_reports=False, use_combined=False, use_features=False)), # input path to accelerometer data)
    ('low_pass_filter', LowPassFilter(cutoff_frequency=config["cutoff_frequency"], sampling_rate=config["data_frequency"], order=config["order"])),
    ('scale_xyz_data', ScaleXYZData(scaler_type=config["scaler_type"])),
    ('extract_features', ExtractFeatures(window_length=config['window_length'], window_step_size=config["window_step_size"], data_frequency=config["data_frequency"],
                                          selected_domains=config['selected_domains'], include_magnitude=config['include_magnitude'])),
    ('classify_movement_data', ClassifyMovementData()),
])

# This will measure the time taken to run the pipeline
start_time = time.time()

# This will start the pipeline and return the classified dataframe
output_df = analyzing_data_pipeline.fit_transform(None)


end_time = time.time()
print(f"Time taken: {int((end_time - start_time) // 60)} minutes and {(end_time - start_time) % 60:.2f} seconds")
