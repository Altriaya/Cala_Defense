import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class PCADetector:
    def __init__(self, n_components=0.95):
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()

    def fit(self, data):
        # data: (N_samples, Features)
        # Flatten time series: (B, T, N, C) -> (B, T*N*C)
        flat_data = data.reshape(data.shape[0], -1)
        self.scaler.fit(flat_data)
        norm_data = self.scaler.transform(flat_data)
        self.pca.fit(norm_data)

    def score(self, data):
        flat_data = data.reshape(data.shape[0], -1)
        norm_data = self.scaler.transform(flat_data)
        
        # Reconstruction Error
        transformed = self.pca.transform(norm_data)
        reconstructed = self.pca.inverse_transform(transformed)
        error = np.mean((norm_data - reconstructed) ** 2, axis=1)
        return error

class IForestDetector:
    def __init__(self, contamination=0.1):
        self.iforest = IsolationForest(contamination=contamination, random_state=42)

    def fit(self, data):
        flat_data = data.reshape(data.shape[0], -1)
        self.iforest.fit(flat_data)

    def score(self, data):
        flat_data = data.reshape(data.shape[0], -1)
        # Decision function: lower is more abnormal. 
        # We want Score: higher is more abnormal.
        # Sklearn decision_function returns negative for outliers (e.g. -0.5) and positive for inliers.
        # So we negate it.
        scores = -self.iforest.decision_function(flat_data)
        return scores
