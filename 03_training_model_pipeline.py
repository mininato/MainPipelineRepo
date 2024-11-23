from sklearn.pipeline import Pipeline
from pipeline_classes import ImportData, PCAHandler, TrainModel
from _config import config
import time

# This pipeline trains a model on the feature dataframe and export the model to a pickle file and general information to a json file
training_model_pipeline = Pipeline([
    ('import_data', ImportData(use_accel=False, use_reports=False, use_combined=False, use_features=True)),
    ('pca_handler', PCAHandler(apply_pca=config["apply_pca"], variance=config["pca_variance"])),
    ('train_model', TrainModel(config=config)),
])

# This will measure the time taken to run the pipeline
start_time = time.time()

# This will start the pipeline and return the model and a report
output_df = training_model_pipeline.fit_transform(None)


end_time = time.time()
print(f"Time taken: {int((end_time - start_time) // 60)} minutes and {(end_time - start_time) % 60:.2f} seconds")
