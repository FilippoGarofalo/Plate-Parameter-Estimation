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