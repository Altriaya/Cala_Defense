import torch
import numpy as np

class BlockMaskingStrategy:
    def __init__(self, mask_ratio=0.4, patch_size=4, block_size=4):
        """
        Args:
            mask_ratio (float): Total percentage of patches to mask.
            patch_size (int): The size of each patch (in time steps).
            block_size (int): Number of consecutive patches to mask (Block Masking).
        """
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.block_size = block_size

    def mask(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, C) or (B, N_patches, D)
               We assume x is already patched or raw time series. 
               Let's assume input matches Transformer logic: (B, Num_Patches, Patch_Dim)
        
        Returns:
            x_masked: (B, Num_Patches, Patch_Dim) - Masked input (zeros or token)
            mask: (B, Num_Patches) - Binary mask (1 for masked, 0 for kept)
            keep_indices: (B, L_keep) - Indices of kept patches (for gather)
        """
        B, L, D = x.shape
        
        num_masked = int(L * self.mask_ratio)
        # Ensure num_masked is multiple of block_size for simplicity
        # or we just try to satisfy ratio.
        
        mask = torch.zeros(B, L, device=x.device)
        
        for b in range(B):
            # We want to place 'blocks' of masks.
            # Number of blocks roughly needed
            n_blocks = max(1, num_masked // self.block_size)
            
            # Simple random block placement
            # Generate more potential start points than needed and pick
            possible_starts = torch.randperm(L - self.block_size + 1, device=x.device)
            
            masked_count = 0
            for start_idx in possible_starts:
                if masked_count >= num_masked:
                    break
                
                # Check if overlapping with existing mask (optional, but good for distribution)
                # Here we just overwrite.
                
                end_idx = start_idx + self.block_size
                if torch.sum(mask[b, start_idx:end_idx]) == 0: # Try to avoid overlap
                    mask[b, start_idx:end_idx] = 1
                    masked_count += self.block_size
        
        # In case we missed too many due to overlap, fill randomly
        current_masked = mask.sum(dim=1)
        # This is a simplified logic. For strict block masking, we might accept slightly less ratio.
        
        mask = mask.bool() # (B, L)
        
        # Apply mask
        # Usually in MAE, we drop the masked tokens or replace with learnable token.
        # But 'PhysicsMAE' might just zero them out if we use standard Transformer, 
        # or use gather/scatter if we use ViT-MAE style.
        # Let's keep the shape (B, L, D) and zero out for simplicity in this demo, 
        # as strict dropping requires positional embeddings management.
        
        x_masked = x.clone()
        x_masked[mask] = 0 # Zero out masked patches
        
        return x_masked, mask

if __name__ == "__main__":
    # Test
    # Batch=2, Length=24 timesteps. 
    # If Patch=1, then L=24.
    x = torch.randn(2, 24, 4) 
    strategy = BlockMaskingStrategy(mask_ratio=0.5, patch_size=1, block_size=6)
    x_masked, mask = strategy.mask(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Masked sample 0:\n{mask[0]}")
