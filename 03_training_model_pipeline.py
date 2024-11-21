from sklearn.pipeline import Pipeline
from pipeline_classes import ImportData, PCAHandler, TrainModel
from _config import config
import time

# # Training model pipeline part (takes features dataframe as input)
training_model_pipeline = Pipeline([
    ('import_data', ImportData(features_data_path="/Users/anhducduong/Documents/GitHub/MainPipelineRepo/features_window_60_step_20_all_features.csv")),
    ('pca_handler', PCAHandler(apply_pca=config["apply_pca"], variance=config["pca_variance"])),
    ('train_model', TrainModel(config=config)),
])

# Run training_model_pipeline
start_time = time.time()
output_df = training_model_pipeline.fit_transform(None)
end_time = time.time()
print(f"Time taken: {int((end_time - start_time) // 60)} minutes and {(end_time - start_time) % 60:.2f} seconds")
