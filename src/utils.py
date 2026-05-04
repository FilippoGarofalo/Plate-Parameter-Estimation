import torch
import torchaudio
import os
import numpy as np


def load_challenge_npz(filepath, device='cpu', dtype=torch.float64):
    data = np.load(filepath)
    
    waveform_np = data['ir'] 
    
    waveform = torch.from_numpy(waveform_np).to(device=device, dtype=dtype)
    
    return waveform.squeeze()

def load_target_audio(filepath: str, target_sr: int = 44100, device: torch.device = torch.device('cpu'),
                      dtype: torch.dtype = torch.float64, normalize: bool = True) -> torch.Tensor:
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File non trovato: {filepath}")

    waveform, sr = torchaudio.load(filepath, backend='soundfile')
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # Resampling
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    
    waveform = waveform.squeeze(0)

    if normalize:
        peak = torch.max(torch.abs(waveform)) + 1e-8
        waveform = waveform / peak

    return waveform.to(device=device, dtype=dtype)


def inverse_softplus(y): 
    # If y is large, the inverse is practically just y itself
    if y > 20.0:
        return float(y)
    # Otherwise, do the normal inverse math safely
    return float(np.log(np.exp(y) - 1.0))

def inverse_tanh(y, min_val, max_val):
    norm_y = (y - min_val) / (max_val - min_val)  # → [0, 1]
    
    norm_y_mapped = 2.0 * norm_y - 1.0
    
    # Clip to avoid infinity
    norm_y_mapped = np.clip(norm_y_mapped, -0.999999, 0.999999)
    
    return float(0.5 * np.log((1 + norm_y_mapped) / (1 - norm_y_mapped)))

import numpy as np

def inverse_map_range_linear(y, min_v, max_v, scale=1.0):
    """Inverse function for map_range_linear"""
    norm_y = (y - min_v) / (max_v - min_v)
    
    norm_y_mapped = 2.0 * norm_y - 1.0
    
    norm_y_mapped = np.clip(norm_y_mapped, -0.999999, 0.999999)
    
    x_raw = np.arctanh(norm_y_mapped) / scale
    
    return float(x_raw)

def inverse_map_range_log(y, min_v, max_v, scale=1.0):
    """Inverse function for map_range_log"""
    log_y = np.log10(y)
    log_min = np.log10(min_v)
    log_max = np.log10(max_v)
    
    norm_y = (log_y - log_min) / (log_max - log_min)
    
    norm_y_mapped = 2.0 * norm_y - 1.0
    
    norm_y_mapped = np.clip(norm_y_mapped, -0.999999, 0.999999)
    
    x_raw = np.arctanh(norm_y_mapped) / scale
    
    return float(x_raw)


def invert_composite_parameters(mu: float, D_over_mu: float, T0_over_mu: float, rho: float, nu: float = 0.25):
    """
    Converts the composite parameters learned by the model back to physical properties.
    
    Args:
        mu (float): Area mass density (learned)
        D_over_mu (float): Bending stiffness to mass ratio (learned)
        T0_over_mu (float): Tension to mass ratio (learned)
        rho (float): Assumed material volume density (kg/m^3)
        nu (float): Poisson's ratio (fixed, default 0.25)
        
    Returns:
        tuple: (h, E, T0) 
               h  = plate thickness (m)
               E  = Young's Modulus (Pa)
               T0 = Applied tension (N/m)
    """
    
    # 1. Recover thickness (h)
    h = mu / rho
    
    # 2. Recover absolute Tension (T0)
    T0 = T0_over_mu * mu
    
    # 3. Recover absolute Bending Stiffness (D)
    D = D_over_mu * mu
    
    # 4. Recover Young's Modulus (E)
    E = (D * 12 * (1 - nu**2)) / (h**3)
    
    return h, E, T0
