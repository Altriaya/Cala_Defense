import numpy as np
import torch

class EnvironmentSplitter:
    def __init__(self, num_envs=3):
        """
        Splits time-series data into environments (e.g. by time of day).
        For PEMS (usually 5-min intervals, 288 steps/day).
        """
        self.num_envs = num_envs
    
    def split(self, data):
        """
        Simulate splitting by time index.
        Real PEMS data usually implies time by index in the daily cycle.
        
        Args:
            data: (B, T, N, C)
        Returns:
            envs: List of datasets/tensors corresponding to different environments.
        """
        # For simplicity in this mock/generic version, we split the batch randomly
        # or assuming the batch is ordered by time.
        # Let's assume we want to split by "Context". 
        # IN REALITY: We should look at timestamp.
        # HERE: We will split the dataset into chunks.
        
        n_samples = data.shape[0]
        chunk_size = n_samples // self.num_envs
        
        envs = []
        for i in range(self.num_envs):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < self.num_envs - 1 else n_samples
            envs.append(data[start:end])
            
        return envs

if __name__ == "__main__":
    data = np.random.randn(100, 24, 10, 4)
    splitter = EnvironmentSplitter(num_envs=3)
    envs = splitter.split(data)
    for i, e in enumerate(envs):
        print(f"Env {i} shape: {e.shape}")
