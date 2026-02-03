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
    def __init__(self, data_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Setup Data
        raw_dataset = MockPEMSDataset(num_samples=200, seq_len=24, num_nodes=10)
        raw_data = raw_dataset.get_all_data()
        
        # Profile & Augment
        profiler = PEMSDataProfiler(raw_data)
        self.coef, self.intercept = profiler.analyze_physics()
        augmenter = PhysicalAugmenter(self.coef, self.intercept)
        
        aug_data_np = augmenter.augment(raw_data)
        
        # 2. Normalization (CRITICAL for Residual Channel sensitivity)
        # Compute mean/std over (N, T, Nodes) for each channel
        self.mean = np.mean(aug_data_np, axis=(0, 1, 2), keepdims=True)
        self.std = np.std(aug_data_np, axis=(0, 1, 2), keepdims=True)
        self.std[self.std < 1e-6] = 1.0 # Avoid div by zero
        
        print(f"Data Stats (Mean): {self.mean.flatten()}")
        print(f"Data Stats (Std):  {self.std.flatten()}")
        
        # Normalize
        norm_data_np = (aug_data_np - self.mean) / self.std
        
        # Reshape for Training: (N*Nodes, T, C)
        N, T, Nodes, C = norm_data_np.shape
        train_data = norm_data_np.transpose(0, 2, 1, 3).reshape(N*Nodes, T, C)
        
        self.train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data).float())
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        
        # 3. Setup Model
        self.model = PhysicsMAE(seq_len=24, in_channels=4, patch_size=1, mask_ratio=0.4, block_size=6).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
    def train(self, epochs=5):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in self.train_loader:
                x = batch[0].to(self.device) # (B, T, C)
                
                pred, mask, _ = self.model(x)
                loss = self.model.compute_loss(pred, x, mask)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            
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
