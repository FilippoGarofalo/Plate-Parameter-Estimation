import torch
import torchaudio

def load_target_audio(filepath: str, target_sr: int = 44100, device: str = 'cpu') -> torch.Tensor:
    """
    Loads a .wav file, converts it to mono, resamples if necessary, 
    and returns it as a 1D PyTorch tensor.
    """
    waveform, sr = torchaudio.load(filepath)
    
    # Convert to mono if it's stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # Resample if the sample rate doesn't match the challenge specs
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        
    # Remove the channel dimension so it is strictly a 1D tensor [num_samples]
    waveform = waveform.squeeze(0)
    
    # Normalize the target audio to [-1, 1] to match our synthesis assumptions
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        waveform = waveform / (max_val + 1e-8)
        
    return waveform.to(device)