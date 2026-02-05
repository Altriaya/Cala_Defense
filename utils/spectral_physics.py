import numpy as np
import torch
import torch.fft
from scipy.ndimage import uniform_filter1d

class SpectralProfiler:
    def __init__(self, seq_len=24):
        self.seq_len = seq_len
        
    def fit(self, data):
        """
        Standard SR is unsupervised/instance-wise, so fit mainly validates data.
        """
        print(f"[Spectral] Initialized Universal Spectral Residual Head.")
        
    def get_residual(self, x_window):
        """
        Calculates Spectral Residual Saliency Map.
        Args:
           x_window: (T, N) usually.
        """
        # Ensure numpy
        if torch.is_tensor(x_window):
            x = x_window.cpu().numpy()
        else:
            x = x_window
            
        if x.ndim == 1: x = x[:, np.newaxis]
        
        T, N = x.shape
        
        # 1. FFT along Time Axis (axis=0)
        fft_x = np.fft.fft(x, axis=0)
        mag = np.abs(fft_x)
        phase = np.angle(fft_x)
        
        # 2. Log Amplitude
        log_amp = np.log(mag + 1e-9)
        
        # 3. Spectral Residual = LogAmp - Smoothed(LogAmp)
        # Smoothing kernel size often 3 or 5
        avg_log_amp = uniform_filter1d(log_amp, size=3, axis=0)
        spectral_residual = log_amp - avg_log_amp
        
        # 4. Saliency = IFFT(exp(SR + i*Phase))
        sr_complex = np.exp(spectral_residual + 1j * phase)
        saliency = np.abs(np.fft.ifft(sr_complex, axis=0))
        
        return saliency # (T, N)

class SpectralAugmenter:
    def __init__(self, profiler):
        self.profiler = profiler
        
    def enforce_consistency(self, batch_norm, mean, std):
        """
        Differentiable Consistency Enforcement (PyTorch).
        Recalculates Channel 1 (Saliency) from Channel 0 (Flow).
        
        Args:
            batch_norm: (B, T, N, C) Normalized Data.
            mean: (1, 1, 1, C)
            std: (1, 1, 1, C)
        """
        # 1. Denormalize Flow (Channel 0)
        # batch_norm is (B, T, N, C) where C>=2. Channel 0 is Flow/Load.
        flow_norm = batch_norm[..., 0]
        mu = mean[..., 0]
        sigma = std[..., 0]
        
        flow_raw = flow_norm * sigma + mu
        
        if torch.is_tensor(flow_raw):
            # Shape (B, T, N)
            
            # 2. FFT along Time Axis (dim=1)
            # PyTorch FFT
            fft_x = torch.fft.fft(flow_raw, dim=1)
            mag = torch.abs(fft_x)
            phase = torch.angle(fft_x)
            log_amp = torch.log(mag + 1e-9)
            
            # 3. Smooth Log Amp (Avg Pooling)
            # We treat (B, N) as batch/channel dims?
            # We want to smooth along T (dim=1).
            # Rearrange for Conv1d: (Batch, Channel, Length) -> (B*N, 1, T)
            B, T, N = flow_raw.shape
            log_amp_reshaped = log_amp.permute(0, 2, 1).reshape(B*N, 1, T)
            
            # Avg Pool 1D (kernel=3, stride=1, padding=1)
            avg_log_amp_flat = torch.nn.functional.avg_pool1d(log_amp_reshaped, kernel_size=3, stride=1, padding=1)
            avg_log_amp = avg_log_amp_flat.reshape(B, N, T).permute(0, 2, 1)
            
            # Residual
            res = log_amp - avg_log_amp
            
            # 4. IFFT
            sr_complex = torch.exp(torch.complex(res, phase))
            saliency = torch.abs(torch.fft.ifft(sr_complex, dim=1)) # (B, T, N)
            
        else:
            # Fallback
            saliency = self.profiler.get_residual(flow_raw)
            
        # 5. Normalize new Saliency
        # We need stats for Channel -1 (Saliency). Usually the last channel.
        # But wait, main.py appends saliency as the LAST channel.
        # If input has 2 channels (Flow, Saliency), it's index 1.
        # If input has 5 channels (Flow, Occ, Speed, PhysRes, Saliency), it's index 4.
        # Let's assume the last channel is meant to be Saliency.
        
        target_channel_idx = batch_norm.shape[-1] - 1
        mu_s = mean[..., target_channel_idx]
        sigma_s = std[..., target_channel_idx]
        
        if torch.is_tensor(flow_raw) and not torch.is_tensor(mu_s):
            mu_s = torch.tensor(mu_s).to(flow_raw.device)
            sigma_s = torch.tensor(sigma_s).to(flow_raw.device)
            
        saliency_norm = (saliency - mu_s) / sigma_s
        
        # 6. Re-assemble: Replace the last channel
        if torch.is_tensor(batch_norm):
            # Separate existing channels
            channels = torch.split(batch_norm, 1, dim=-1)
            # channels is tuple of (B, T, N, 1) tensors
            
            new_channels = list(channels)
            saliency_norm_exp = saliency_norm.unsqueeze(-1) # (B, T, N, 1)
            new_channels[-1] = saliency_norm_exp
            
            out = torch.cat(new_channels, dim=-1)
        else:
            out = batch_norm.copy()
            out[..., -1] = saliency_norm
            
        return out
