import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import sys
import os

# Adjust path to import utils
curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
sys.path.append(parent_dir)

from utils.mock_data import MockPEMSDataset
from utils.data_profiler import PEMSDataProfiler
from utils.physical_augmenter import PhysicalAugmenter
from consistency.mae_model import PhysicsMAE

class MAETrainer:
    def __init__(self, model=None, device=None, data_path=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        
        # If model is not provided, initialize standard one (Legacy mode)
        if self.model is None:
             # Legacy init logic...
             pass 
        else:
             self.model = self.model.to(self.device)
             self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def train_model(self, loader, epochs=5):
        """
        Train using external DataLoader.
        """
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                x = batch[0].to(self.device) # (B, T, C)
                # print(f"DEBUG Trainer Input Shape: {x.shape}") 
                pred, mask, _ = self.model(x)
                loss = self.model.compute_loss(pred, x, mask)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if epoch % 2 == 0:
                print(f"    [MAE Epoch {epoch}] Loss: {total_loss/len(loader):.4f}")

    # Legacy train method kept for backward compatibility if needed, 
    # but we primarily use train_model now.
    def train(self, epochs=5):
        pass
            
    def verify_masking_sensitivity(self):
        """
        Verify that Block Masking makes reconstruction harder for triggers compared to random noise.
        """
        print("\n[Verification] Testing sensitivity to Trigger...")
        self.model.eval()
        
        # Create a sample (Raw)
        # 1. Clean
        raw_clean = np.random.randn(1, 24, 1, 3).astype(np.float32) # Dummy content
        # Make it physically consistent-ish? 
        # Actually proper way: use MockData generate a single clean sample
        mock = MockPEMSDataset(num_samples=1, num_nodes=1)
        raw_sample = mock.get_all_data() # (1, 24, 1, 3)
        
        # Augment
        augmenter = PhysicalAugmenter(self.coef, self.intercept)
        aug_sample = augmenter.augment(raw_sample) # (1, 24, 1, 4)
        
        # Normalize
        norm_clean = (aug_sample - self.mean) / self.std
        clean_tensor = torch.from_numpy(norm_clean[...,0,:]).to(self.device) # (1, 24, 4)

        # 2. Trigger
        # Inject trigger into RAW data then augment/normalize
        # Inject into Flow
        trigger_raw = raw_sample.copy()
        t = np.linspace(0, 2*np.pi, 24)
        sine_wave = np.sin(t) * 20 + 200 # Add large sine wave to Flow (typically ~250)
        # Replacing flow with sine wave.
        trigger_raw[0, :, 0, 0] = sine_wave

        # Re-calc residual
        aug_trigger = augmenter.augment(trigger_raw)
        
        # Normalize with SAME stats
        norm_trigger = (aug_trigger - self.mean) / self.std
        trigger_tensor = torch.from_numpy(norm_trigger[...,0,:]).to(self.device)

        with torch.no_grad():
            # Clean Pred
            pred_c, mask_c, _ = self.model(clean_tensor)
            loss_c = self.model.compute_loss(pred_c, clean_tensor, mask_c)
            
            # Trigger Pred
            pred_t, mask_t, _ = self.model(trigger_tensor)
            loss_t = self.model.compute_loss(pred_t, trigger_tensor, mask_t)
            
        print(f"Clean Reconstruction Loss: {loss_c.item():.6f}")
        print(f"Trigger Reconstruction Loss: {loss_t.item():.6f}")
        ratio = loss_t.item() / (loss_c.item() + 1e-6)
        print(f"Ratio (Trigger/Clean): {ratio:.2f}")
        
        if ratio > 2.0:
            print("SUCCESS: Trigger detected via high reconstruction error!")
        else:
            print("WARNING: Trigger not distinguished enough.")

if __name__ == "__main__":
    trainer = MAETrainer()
    trainer.train(epochs=10)
    trainer.verify_masking_sensitivity()
