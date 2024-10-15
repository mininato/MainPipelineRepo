# Configuration for default settings
config = {
    # Configuration for data import
    "time_window": 2,  # Time window in minutes
    "scaler_type": "standard",  # 'standard' or 'minmax'
    "bin_size_minutes": 3,  # Bin size in minutes

    # Configuration for feature extraction
    "window_length": 120,  # Window length in seconds
    "window_step_size": 120,  # Step size in seconds
    "data_frequency": 25,  # Data frequency in Hz
    "selected_domains": None,  # Default: Every domain. Available: ['time_domain', 'spatial', 'frequency', 'statistical', 'wavelet']
    "include_magnitude": True,  # Whether to include magnitude-based features or not

    # Configuration for PCA
    "apply_pca": False,  # Whether to apply PCA or not
    "pca_variance": 0.95,  # PCA variance threshold

    # Configuration for model training
    "classifier": "xgboost",  # Default classifier ('xgboost', 'svm', 'randomforest')
    "target": "combined",  # Target to train ('combined', 'arousal', 'valence')

    # Configuration for hyperparameter tuning
    "n_splits": 5,  # Number of splits for cross-validation
    "n_iter": 30,  # Number of iterations for hyperparameter optimization
    "n_jobs": -1,  # Number of jobs for parallel processing
    "n_points": 1,  # Number of parameter configurations to evaluate at each iteration

    # Parameter space for hyperparameter tuning (if not None, custom parameter space will be used)
    "param_space": None,  # Set to None to use default inside the TrainModel class
}
