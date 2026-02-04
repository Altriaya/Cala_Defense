import numpy as np
import torch

class PhysicalAugmenter:
    def __init__(self, coef=None, v_f=None, k_j=None):
        """
        Args:
            coef: Linear coefficient q = C * v * k
            v_f: Free flow speed (Greenshields)
            k_j: Jam density (Greenshields)
            
        Mode:
            If v_f and k_j are provided, uses Non-linear Greenshields Residual:
              Res = Flow - v_f * Density * (1 - Density/k_j)
              (Assuming Density approx Occupancy for scale-invariant check, or normalized)
            Else, uses Linear Residual:
              Res = Flow - Coef * Speed * Occupancy
        """
        self.coef = coef
        self.v_f = v_f
        self.k_j = k_j
        
        if self.v_f is not None and self.k_j is not None:
            print(f"[Augmenter] Using Non-Linear Greenshields (vf={v_f:.1f}, kj={k_j:.2f})")
        elif self.coef is not None:
             print(f"[Augmenter] Using Linear Physics (C={self.coef:.2f})")
        else:
            # Fallback
            self.coef = 1.0
            print(f"[Augmenter] Using Linear Physics (C={self.coef:.2f}) as no specific model params provided.")
    
    def augment(self, data):
        """
        Appends Physical Residual Channel.
        Args:
            data: (B, T, N, 3) 
            Channels: 0:Flow, 1:Occ, 2:Speed (Standardized PEMS)
        Returns:
            aug_data: (B, T, N, 4) [Flow, Occ, Speed, Residual]
        """
        # Extract channels based on the new standard: 0:Flow, 1:Occ, 2:Speed
        flow = data[..., 0]
        occ = data[..., 1] 
        speed = data[..., 2]
        
        if isinstance(data, torch.Tensor):
            if self.v_f is not None and self.k_j is not None:
                # Non-linear Greenshields Residual
                # q_model = v_f * k * (1 - k/k_j)
                q_pred = self.v_f * occ * (1 - occ / self.k_j)
                residual = flow - q_pred
            else:
                # Linear Residual
                residual = flow - self.coef * (speed * occ)
            
            aug_data = torch.cat([data, residual.unsqueeze(-1)], dim=-1)
            return aug_data
            
        elif isinstance(data, np.ndarray):
            # Same channel extraction: 0:Flow, 1:Occ, 2:Speed
            flow = data[..., 0]
            occ = data[..., 1] 
            speed = data[..., 2]
            
            if self.v_f is not None and self.k_j is not None:
                q_pred = self.v_f * occ * (1 - occ / self.k_j)
                residual = flow - q_pred
            else:
                residual = flow - self.coef * (speed * occ)
            
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
