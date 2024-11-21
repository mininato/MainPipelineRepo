# Configuration for default settings
config = {
    # Paths for data import
    "accel_path": "/Users/anhducduong/Documents/GitHub/MainPipelineRepo/AccelerometerMeasurements_backup.csv",  # Path to the accelerometer data
    "reports_path": "/Users/anhducduong/Documents/GitHub/MainPipelineRepo/Cleaned_SelfReports.csv",  # Path to the self-reports data
    "combined_data_path": "/Users/anhducduong/Documents/GitHub/MainPipelineRepo/combined_data_timewindow_2min_labels_valence_arousal.csv",  # Path to the combined data
    "features_data_path": "/Users/anhducduong/Documents/GitHub/MainPipelineRepo/features_window_60_step_20_all_features.csv",  # Path to the features data
    "model_path": "/Users/anhducduong/Documents/GitHub/MainPipelineRepo/xgboost_best_model_arousal.pkl",  # Path to the trained model
    
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
    "window_length": 60,  # Window length in seconds / 60
    "window_step_size": 20,  # Step size in seconds / 10%-50% of window_length / 20
    "data_frequency": 25,  # Data frequency in Hz
    "selected_domains": None,  # Default: Every domain / 'time_domain', 'spatial', 'frequency', 'statistical', 'wavelet' / multiple domains: ["time_domain", "frequency"] / order is not important
    "include_magnitude": True,  # Include magnitude-based features or not

    #Configuration for Low-pass filter
    "cutoff_frequency": 10,  # Cut-off frequency for the low-pass filter
    "order": 4,  # Order of the filter

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
    "param_space": {
        "learning_rate": (0.05, 0.2), 
        "n_estimators": (200, 800),
        "max_depth": (4, 8),
        "min_child_weight": (1, 5),
        "subsample": (0.6, 0.9),
        "colsample_bytree": (0.6, 0.9),
        "gamma": (0, 5),
        "reg_alpha": (0, 5),
        "reg_lambda": (0, 5)
    },  # Set to None to use default inside the TrainModel class
}