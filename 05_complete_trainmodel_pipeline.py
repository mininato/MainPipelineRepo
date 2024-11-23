from pipeline_classes import ImportData, LowPassFilter, ScaleXYZData, ExtractFeatures, CreateCombinedDataFrame, TrainModel, PCAHandler
from _config import config
from sklearn.pipeline import Pipeline
import time

# This is the complete pipeline that will be used to train a model on the combined dataframe and export the model to a pickle file and general information to a json file
complete_training_model_pipeline = Pipeline([
    ('import_data', ImportData(use_accel=True, use_reports=True, use_combined=False, use_features=False)),
    ('create_combined_dataframe', CreateCombinedDataFrame(time_window=config["time_window"], label_columns=config["label_columns"])),
    ('low_pass_filter', LowPassFilter(cutoff_frequency=config["cutoff_frequency"], sampling_rate=config["data_frequency"], order=config["order"])),
    ('scale_xyz_data', ScaleXYZData(scaler_type=config["scaler_type"])),
    ('extract_features', ExtractFeatures(window_length=config["window_length"],
                                         window_step_size=config["window_step_size"],
                                         data_frequency=config["data_frequency"],
                                         selected_domains=config["selected_domains"],
                                         include_magnitude=config["include_magnitude"],
                                         label_columns=config["label_columns"])),
    ('pca_handler', PCAHandler(apply_pca=config["apply_pca"], variance=config["pca_variance"])),
    ('train_model', TrainModel(config=config)),
])

# This will measure the time taken to run the pipeline
start_time = time.time()

# This will start the pipeline and return the model and a report
output_df = complete_training_model_pipeline.fit_transform(None)


end_time = time.time()
print(f"Time taken: {int((end_time - start_time) // 60)} minutes and {(end_time - start_time) % 60:.2f} seconds")