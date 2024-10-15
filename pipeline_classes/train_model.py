import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedGroupKFold
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib
from skopt.space import Real, Integer, Categorical
from sklearn.metrics import classification_report, accuracy_score
import json
from sklearn.preprocessing import LabelEncoder
from _config import config

class TrainModel(BaseEstimator, TransformerMixin):
    def __init__(self, config):
        self.config = config
        self.target = config.get("target", "combined")  # Default to "combined"
        self.label_encoder = LabelEncoder()
        self.selected_domains = self.config.get("selected_domains", "All domains")  # Default to "All domains" if None

    def get_default_param_space(self, classifier):
        # """
        # Returns the default hyperparameter space for a given classifier.
        # """
        if classifier == 'xgboost':
            return {
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'n_estimators': Integer(100, 1000),
                'max_depth': Integer(3, 10),
                'min_child_weight': (1, 10),
                'subsample': (0.5, 1.0),
                'colsample_bytree': (0.5, 1.0),
                'gamma': (0, 10),
                'reg_alpha': (0, 10),  # alpha in xgboost is reg_alpha
                'reg_lambda': (0, 10), # lambda in xgboost is reg_lambda
            }
        elif classifier == 'svm':
            return {
                'C': Real(0.1, 10, prior='log-uniform'),
                'kernel': Categorical(['linear', 'rbf'])
            }
        elif classifier == 'randomforest':
            return {
                'n_estimators': Integer(100, 1000),
                'max_depth': Integer(3, 10)
            }
        else:
            raise ValueError(f"Unsupported classifier type: {classifier}")

    def fit(self, X, y=None):
        # Fit the label encoder on the target column
        print(f"Encoding the target labels for '{self.target}'...")
        self.label_encoder.fit(X[self.target])

        # Print the mapping between original labels and encoded labels
        original_labels = list(self.label_encoder.classes_)
        encoded_labels = list(range(len(original_labels)))
        label_mapping = dict(zip(encoded_labels, original_labels))
        print(f"Label encoding complete. Mapping: {label_mapping}")

        # Transform the target column and add it as 'encoded_target'
        X['encoded_target'] = self.label_encoder.transform(X[self.target])

        # Value counts for the encoded target        
        value_counts = X['encoded_target'].value_counts().to_dict()
        print(f"Value counts for encoded target: {value_counts}")
        
        # Pop everything except the features and the encoded target (groupid,arousal,valence,context,participantId,combined)
        groups = X.pop('groupid')
        arousal = X.pop('arousal')
        valence = X.pop('valence')
        context = X.pop('context')
        participantId = X.pop('participantId')
        combined = X.pop('combined')
        
        # pop the encoded target as Y
        y = X.pop('encoded_target')

        # Store the feature names for later use
        feature_names = X.columns.tolist()

        # Choose classifier
        classifier = self.config['classifier']
        if classifier == 'xgboost':
            model = XGBClassifier(objective='multi:softmax', random_state=42)
        elif classifier == 'svm':
            model = SVC(probability=True)
        elif classifier == 'randomforest':
            model = RandomForestClassifier(random_state=42)
        else:
            raise ValueError(f"Unsupported classifier type: {classifier}")

        print(f"Training the model using {classifier}...")

        # Use user-defined param_space if provided, otherwise use default
        print(f"Classifier: {classifier}")
        default_param_space = self.get_default_param_space(classifier)
        # Use user-defined param_space if provided, otherwise use default
        param_space = self.config.get("param_space") or default_param_space     

        # Hyperparameter tuning using Bayesian optimization
        sgkf = StratifiedGroupKFold(n_splits=self.config['n_splits'])
        print(f"Parameter space being used: {param_space}")
        if param_space is None:
            raise ValueError("Parameter space cannot be None. Please check the classifier configuration.")

        opt = BayesSearchCV(
            estimator=model,
            search_spaces=param_space,
            cv=sgkf,
            n_iter=self.config['n_iter'],
            n_jobs=self.config['n_jobs'],
            n_points=self.config['n_points'],
            verbose=1,
            scoring='accuracy'
        )

        print("Hyperparameter tuning in progress...")

        # Fit the model using the encoded target
        opt.fit(X, y, groups=groups)
        self.best_model = opt.best_estimator_
        print(f"Best parameters found: {opt.best_params_}")

        #DONE print metrics and feature importance
        # Print classification metrics
        y_pred = self.best_model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, target_names=self.label_encoder.classes_, output_dict=True)
        
        classification_report_json = report  
        with open('classification_report.json', 'w') as f:
            json.dump(classification_report_json, f, indent=4)

        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{report}")

        # Saving Part
        # Save the best model
        model_name = f"{classifier}_best_model_{self.target}.pkl"
        joblib.dump(self.best_model, model_name)
        print("Model saved successfully.")

        # Save metrics and feature importance
        model_metadata = {
            "best_params": opt.best_params_,
            "accuracy": accuracy,
            "classification_report": classification_report_json,
            "label_mapping": label_mapping,
            "model_name": model_name,
            "value_counts": value_counts,
            "selected_domains": self.selected_domains,  
            "include_magnitude": self.config.get("include_magnitude", True)  
            #TODO Confusion Matrix hinzufügen      
            }

        if hasattr(self.best_model, "feature_importances_"):
            feature_importances = self.best_model.feature_importances_
            # Convert feature importances to native Python floats
            feature_importance_dict = {feature: float(importance) for feature, importance in zip(feature_names, feature_importances)}
            model_metadata["feature_importances"] = feature_importance_dict
            print("Feature Importances:")
            for feature, importance in feature_importance_dict.items():
                print(f"{feature}: {importance:.4f}")


        metadata_file = f"{classifier}_model_metadata_{self.target}.json"

        with open(metadata_file, "w") as f:
            json.dump(model_metadata, f, indent=4)
            print(f"Model metadata saved to {metadata_file}.")

        return self

    def transform(self, X):
        return X  # Placeholder for transform step (not needed for training)