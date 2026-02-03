import numpy as np
import torch
from torch.utils.data import Dataset

class MockPEMSDataset(Dataset):
    def __init__(self, num_samples=100, num_nodes=20, seq_len=24, noise_level=0.05):
        """
        Generates synthetic traffic data following physical laws: Flow = Speed * Occupancy.
        
        Args:
            num_samples: Number of samples.
            num_nodes: Number of sensors.
            seq_len: Time sequence length.
            noise_level: Noise magnitude.
        """
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        
        # 1. Generate Speed (v): modeled as smooth waves (e.g., free flow to congestion)
        # Range: 20 to 70 mph
        self.speed = np.random.uniform(20, 70, size=(num_samples, seq_len, num_nodes))
        
        # 2. Generate Occupancy (rho): Inverse relation to speed (roughly)
        # High speed -> Low occupancy, Low speed -> High occupancy
        # Model: rho \propto 1/v + noise
        self.occupancy = (1000 / self.speed) + np.random.normal(0, 2, size=self.speed.shape)
        self.occupancy = np.clip(self.occupancy, 5, 95) / 100.0 # Normalize to 0-1
        
        # 3. Generate Flow (q): q = Speed * Map(Occupancy) * C
        # Fundamental Diagram: q = v * k, where k is density. Occupancy is proxy for k.
        # Let's assume q = v * occupancy * 1000 (just a scaling factor)
        self.flow = self.speed * self.occupancy * 10 + np.random.normal(0, noise_level*10, size=self.speed.shape)
        self.flow = np.maximum(self.flow, 0)
        
        # Stack: (B, T, N, 3) -> [Flow, Speed, Occupancy]
        # PEMS usually: Flow, Occupancy, Speed. Let's stick to user prompt order or standard.
        # User prompt mentions: Flow/Speed/Occupancy. 
        # Let's standardize on channel order: 0:Flow, 1:Speed, 2:Occupancy
        self.data = np.stack([self.flow, self.speed, self.occupancy], axis=-1).astype(np.float32)
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])

    def get_all_data(self):
        return self.data

if __name__ == "__main__":
    # Test
    dataset = MockPEMSDataset()
    print(f"Data shape: {dataset.data.shape}")
    print(f"Sample mean (Flow, Speed, Occ): {np.mean(dataset.data, axis=(0,1,2))}")
