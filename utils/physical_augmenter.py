import numpy as np
import torch

class PhysicalAugmenter:
    def __init__(self, coef=1.0, intercept=0.0):
        """
        Args:
            coef: The mixing coefficient C derived from profiler.
            intercept: The bias term.
        """
        self.coef = coef
        self.intercept = intercept
    
    def augment(self, data):
        """
        Appends Physical Residual Channel.
        Args:
            data: (B, T, N, 3) [Flow, Speed, Occ]
        Returns:
            aug_data: (B, T, N, 4) [Flow, Speed, Occ, Residual]
        """
        if isinstance(data, torch.Tensor):
            flow = data[..., 0]
            speed = data[..., 1]
            occ = data[..., 2]
            
            # Expected Flow = C * (v * rho) + b
            expected_flow = self.coef * (speed * occ) + self.intercept
            
            # Residual = |Actual - Expected|
            # We take abs because deviation in any direction is unnatural
            # Or we can keep sign if we want directional info. 
            # Given user said "The channel is close to 0", let's use raw difference.
            # But MAE usually likes normalized inputs. 
            # Let's use raw difference for now, the normalizer later handles scaling.
            residual = flow - expected_flow
            
            aug_data = torch.cat([data, residual.unsqueeze(-1)], dim=-1)
            return aug_data
            
        elif isinstance(data, np.ndarray):
            flow = data[..., 0]
            speed = data[..., 1]
            occ = data[..., 2]
            
            expected_flow = self.coef * (speed * occ) + self.intercept
            residual = flow - expected_flow
            
            aug_data = np.concatenate([data, residual[..., np.newaxis]], axis=-1)
            return aug_data
        else:
            raise TypeError("Data must be torch.Tensor or np.ndarray")

if __name__ == "__main__":
    from mock_data import MockPEMSDataset
    from data_profiler import PEMSDataProfiler
    
    # 1. Get Data
    dataset = MockPEMSDataset(num_samples=10)
    raw_data = dataset.get_all_data()
    
    # 2. Profile
    profiler = PEMSDataProfiler(raw_data)
    c, b = profiler.analyze_physics()
    
    # 3. Augment
    augmenter = PhysicalAugmenter(coef=c, intercept=b)
    aug_data = augmenter.augment(raw_data)
    
    print(f"Original Shape: {raw_data.shape}")
    print(f"Augmented Shape: {aug_data.shape}")
    print(f"Mean Residual: {np.mean(np.abs(aug_data[..., 3])):.4f}")
