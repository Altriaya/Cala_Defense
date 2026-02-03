import torch
import torch.nn as nn
import torch.fft

class TimesBlock(nn.Module):
    def __init__(self, in_channels, d_model, k=3):
        super(TimesBlock, self).__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU()
        )

    def forward(self, x):
        # x: [B, C, T]
        B, C, T = x.size()
        
        # 1. FFT to find top-k periods
        xf = torch.fft.rfft(x, dim=-1)
        frequency_list = abs(xf).mean(0).mean(0)
        frequency_list[0] = 0 # Ignore DC
        _, top_list = torch.topk(frequency_list, self.k)
        top_list = top_list.detach().cpu().numpy()
        
        period_list = [int(T / (f + 1e-5)) for f in top_list]
        
        res = []
        for period in period_list:
            # Reshape 1D -> 2D
            if period <= 0: period = 1
            # Padding to make divisible
            pad_len = (period - T % period) % period
            if pad_len > 0:
                xp = torch.nn.functional.pad(x, (0, pad_len))
            else:
                xp = x
            
            # (B, C, L)
            _, _, L = xp.shape
            # (B, C, L//P, P)
            xp = xp.reshape(B, C, L // period, period)
            
            # Conv2D
            out = self.conv(xp)
            
            # Reshape back
            out = out.reshape(B, -1, L)
            out = out[..., :T]
            res.append(out)
            
        # Weighted Variance Sum (Simplified to sum for demo)
        res = torch.stack(res, dim=-1) # (B, d_model, T, k)
        # Softmax on periods? Simplified: Mean
        res = res.mean(-1)
        
        return res

class TimesNetForecaster(nn.Module):
    def __init__(self, in_dim=1, out_dim=12, d_model=32, k=2):
        super(TimesNetForecaster, self).__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.block = TimesBlock(d_model, d_model, k=k)
        # Flatten and project to output
        self.out_proj = nn.Linear(d_model, out_dim)
        
    def forward(self, x):
        """
        x: (B, T, C_in)
        """
        B, T, C = x.shape
        
        # Proj: (B, T, d_model)
        enc = self.input_proj(x)
        
        # TimesBlock expects (B, C, T)
        enc = enc.permute(0, 2, 1) # (B, d, T)
        
        enc_out = self.block(enc) # (B, d, T)
        
        # Output: We want prediction. 
        # TimesNet usually predicts future by extending representations or using last state.
        # Let's simple-pool or take last state.
        
        # (B, d, T) -> (B, d)
        last_state = enc_out[:, :, -1]
        
        # (B, out_dim)
        pred = self.out_proj(last_state)
        
        return pred
