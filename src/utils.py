import torch
import torchaudio
import os
import numpy as np




def load_target_audio(filepath: str, target_sr: int = 44100, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """Loads and preprocesses the target impulse response."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot find target audio at {filepath}")
        
    waveform, sr = torchaudio.load(filepath)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # Resample if necessary
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    
    # Make it 1D
    waveform = waveform.squeeze(0)
    
    # Normalization 
    peak = torch.max(torch.abs(waveform)) + 1e-8
    waveform = waveform / peak
    
    return waveform.to(device)


def inverse_sigmoid(y, min_val, max_val): 
    norm_y = (y - min_val) / (max_val - min_val)
    return np.log(norm_y / (1.0 - norm_y))

def inverse_softplus(y): 
    # If y is large, the inverse is practically just y itself
    if y > 20.0:
        return float(y)
    # Otherwise, do the normal inverse math safely
    return float(np.log(np.exp(y) - 1.0))