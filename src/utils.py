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
        print(f"Resampling from {sr} Hz to {target_sr} Hz")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    
    waveform = waveform.squeeze(0)

    if normalize:
        peak = torch.max(torch.abs(waveform)) + 1e-8
        waveform = waveform / peak

    return waveform.to(device=device, dtype=dtype)


def logit(p):
    """Calcola l'inversa della funzione Sigmoide."""
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return np.log(p / (1.0 - p))

def inverse_map_range_linear(y, min_v, max_v):
    """Inversa di map_range_linear (basata su Sigmoide)"""
    norm_y = (y - min_v) / (max_v - min_v)
    
    x_raw = logit(norm_y)
    
    return float(x_raw)

def inverse_map_range_log(y, min_v, max_v):
    """Inversa di map_range_log (basata su Log Naturale e Sigmoide)"""
    log_y = np.log(y)
    log_min = np.log(min_v)
    log_max = np.log(max_v)
    
    norm_y = (log_y - log_min) / (log_max - log_min)
    
    x_raw = logit(norm_y)
    
    return float(x_raw)
