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