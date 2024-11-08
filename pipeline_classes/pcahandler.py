import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from _config import config
    
class PCAHandler(BaseEstimator, TransformerMixin):
    def __init__(self, apply_pca=False, variance=0.95):
        self.apply_pca = apply_pca
        self.variance = variance
        self.pca = None

    def fit(self, X, y=None):
        if self.apply_pca:
            self.pca = PCA(n_components=self.variance)
            self.pca.fit(X)
        return self

    def transform(self, X):
        if self.apply_pca and self.pca:
            X_transformed = self.pca.transform(X)
            return pd.DataFrame(X_transformed, index=X.index)
        
        return X