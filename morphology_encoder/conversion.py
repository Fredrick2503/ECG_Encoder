import torch
import torch.nn.functional as F
import numpy as np

def ecg_to_gaf(x: torch.Tensor, type: str = 'gasf', target_length: int = 64) -> torch.Tensor:
    """
    Converts 12-lead ECG signals into Gramian Angular Fields.
    
    Args:
        x: Input tensor of shape (..., 12, signal_length)
        type: 'gasf' (Summation) or 'gadf' (Difference)
        target_length: Length to downsample the signal to before computing GAF.
        
    Returns:
        torch.Tensor: GAF tensor of shape (..., 12, target_length, target_length)
    """
    orig_shape = x.shape
    x_flat = x.view(-1, 12, orig_shape[-1])
    x_down = F.interpolate(x_flat, size=target_length, mode='linear', align_corners=False)
    
    min_vals = x_down.min(dim=-1, keepdim=True)[0]
    max_vals = x_down.max(dim=-1, keepdim=True)[0]
    denom = max_vals - min_vals + 1e-8
    x_scaled = 2.0 * (x_down - min_vals) / denom - 1.0
    x_scaled = torch.clamp(x_scaled, -1.0, 1.0)
    
    phi = torch.acos(x_scaled)
    cos_phi = x_scaled
    sin_phi = torch.sin(phi)
    
    cos_phi_i = cos_phi.unsqueeze(-1)
    cos_phi_j = cos_phi.unsqueeze(-2)
    sin_phi_i = sin_phi.unsqueeze(-1)
    sin_phi_j = sin_phi.unsqueeze(-2)
    
    if type.lower() == 'gasf':
        gaf = cos_phi_i * cos_phi_j - sin_phi_i * sin_phi_j
    elif type.lower() == 'gadf':
        gaf = sin_phi_i * cos_phi_j - cos_phi_i * sin_phi_j
    else:
        raise ValueError(f"Unknown GAF type: {type}")
        
    out_shape = list(orig_shape[:-2]) + [12, target_length, target_length]
    return gaf.view(*out_shape)


def ecg_to_spectrogram(x: torch.Tensor, n_fft: int = 64, hop_length: int = 32) -> torch.Tensor:
    """
    Converts 12-lead ECG signals into Magnitude Spectrograms.
    
    Args:
        x: Input tensor of shape (..., 12, signal_length)
        n_fft: FFT window size
        hop_length: FFT hop length
        
    Returns:
        torch.Tensor: Spectrogram magnitude of shape (..., 12, freq_bins, time_steps)
    """
    orig_shape = x.shape
    x_flat = x.view(-1, orig_shape[-1])
    
    window = torch.hann_window(n_fft, device=x.device)
    stft_out = torch.stft(
        x_flat,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
        center=True
    )
    magnitude = torch.abs(stft_out)
    
    freq_bins, time_steps = magnitude.shape[-2:]
    out_shape = list(orig_shape[:-2]) + [12, freq_bins, time_steps]
    return magnitude.view(*out_shape)


def ecg_to_scalogram(x: torch.Tensor, scales: int = 64, wavelet: str = 'morl', target_length: int = 64) -> torch.Tensor:
    """
    Converts 12-lead ECG signals into Scalograms using Continuous Wavelet Transform.
    """
    orig_shape = x.shape
    x_flat = x.view(-1, 12, orig_shape[-1])
    
    if orig_shape[-1] != target_length:
        x_down = F.interpolate(x_flat, size=target_length, mode='linear', align_corners=False)
    else:
        x_down = x_flat
        
    device = x.device
    kernel_size = 31
    t = torch.linspace(-3, 3, kernel_size, device=device)
    scale_arr = torch.linspace(1.0, 10.0, scales, device=device)
    
    kernels_real = []
    kernels_imag = []
    for s in scale_arr:
        t_s = t / s
        w_real = torch.exp(-0.5 * t_s**2) * torch.cos(5.0 * t_s)
        w_imag = torch.exp(-0.5 * t_s**2) * torch.sin(5.0 * t_s)
        w_real = w_real / torch.sqrt(s)
        w_imag = w_imag / torch.sqrt(s)
        kernels_real.append(w_real)
        kernels_imag.append(w_imag)
        
    weight_real = torch.stack(kernels_real).unsqueeze(1)
    weight_imag = torch.stack(kernels_imag).unsqueeze(1)
    
    x_in = x_down.view(-1, 1, target_length)
    pad = kernel_size // 2
    out_real = F.conv1d(x_in, weight_real, padding=pad)
    out_imag = F.conv1d(x_in, weight_imag, padding=pad)
    
    magnitude = torch.sqrt(out_real**2 + out_imag**2)
    out_shape = list(orig_shape[:-2]) + [12, scales, target_length]
    return magnitude.view(*out_shape)
