import torch
import torch.nn as nn
from .masking import BlockMaskingStrategy

class PhysicsMAE(nn.Module):
    def __init__(self, 
                 seq_len=24, 
                 in_channels=4, 
                 embed_dim=64, 
                 depth=4, 
                 num_heads=4, 
                 patch_size=1,
                 mask_ratio=0.5,
                 block_size=4):
        """
        Args:
            in_channels: 3 (F,S,O) + 1 (Residual) = 4
        """
        super().__init__()
        self.patch_size = patch_size
        self.seq_len = seq_len
        self.num_patches = seq_len // patch_size
        self.embed_dim = embed_dim
        
        # Masking
        self.mask_strategy = BlockMaskingStrategy(mask_ratio, patch_size, block_size)
        
        # Embedding
        self.patch_embed = nn.Linear(in_channels * patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Decoder
        # We project back to input dim
        self.decoder_pred = nn.Linear(embed_dim, in_channels * patch_size)
        
    def forward(self, x):
        """
        x: (B, T, C)
        """
        B, T, C = x.shape
        
        # 1. Patchify
        # (B, T, C) -> (B, Num_Patches, Patch_Size*C)
        x_patched = x.view(B, self.num_patches, self.patch_size * C)
        
        # 2. Embed & Add Pos
        x_emb = self.patch_embed(x_patched) + self.pos_embed
        
        # 3. Masking
        # Note: We mask AFTER embedding usually, or mask the input.
        # Let's use the zero-out strategy from masking.py for simplicity
        x_masked, mask = self.mask_strategy.mask(x_emb)
        
        # 4. Encode
        latent = self.encoder(x_masked)
        
        # 5. Decode/Reconstruct
        pred_patched = self.decoder_pred(latent)
        
        # 6. Unpatchify
        pred = pred_patched.view(B, T, C)
        
        return pred, mask, x_patched

    def compute_loss(self, pred, target, mask):
        """
        pred: (B, T, C)
        target: (B, T, C)
        mask: (B, Num_Patches) -> align to T
        """
        # Align mask to (B, T, C)
        # mask is (B, L) where L is num_patches
        B, L = mask.shape
        T = target.shape[1]
        
        # Expand mask to time dimension
        # (B, L, 1) -> (B, L, Patch_Size) -> (B, T)
        mask_expanded = mask.unsqueeze(-1).repeat(1, 1, self.patch_size).view(B, T)
        mask_expanded = mask_expanded.unsqueeze(-1) # (B, T, 1)
        
        # Loss 1: Reconstruction on MASKED parts
        loss_recon = (pred - target) ** 2
        loss_recon = (loss_recon * mask_expanded).sum() / (mask_expanded.sum() + 1e-6)
        
        # Loss 2: Physics Residual Check on ALL parts (or just masked?)
        # User said: "MAE will learn check this channel is 0"
        # The target for residual channel (index 3) is ALREADY in 'target'.
        # If target is clean, residual is ~0.
        # If we reconstruct 'target', we reconstruct 0.
        # So standard MSE covers this.
        # BUT, we can add extra weight to the 4th channel.
        
        return loss_recon
