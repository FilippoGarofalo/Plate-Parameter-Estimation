import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeDomainEnergyLoss(nn.Module):
    """
    Computes a combined Relative Time-Domain MSE, Multi-Scale Spectral (MSS) Loss, 
    and a Low-Pass Magnitude Loss.
    """
    def __init__(self, mse_weight: float = 0.0, stft_weight: float = 1.0, 
                 energy_weight: float = 1.0, lowpass_weight: float = 10.0, 
                 cutoff_hz: float = 200.0, sr: float = 44100,
                 fft_sizes: list = [64, 256, 1024, 4096]):
        super(TimeDomainEnergyLoss, self).__init__()
        self.mse_weight = mse_weight
        self.stft_weight = stft_weight
        self.energy_weight = energy_weight
        self.lowpass_weight = lowpass_weight
        self.cutoff_hz = cutoff_hz
        self.sr = sr
        self.fft_sizes = fft_sizes


    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        pred_audio = pred_audio.squeeze()
        target_audio = target_audio.squeeze()
    
        # --- 1. MSE Loss ---
        target_variance = torch.mean(target_audio ** 2) + 1e-8
        raw_mse = F.mse_loss(pred_audio, target_audio)
        mse_loss = raw_mse / target_variance
        
        # --- 2. Energy Loss ---
        pred_energy = pred_audio ** 2
        target_energy = target_audio ** 2
        
        target_energy_var = torch.mean(target_energy ** 2) + 1e-8
        raw_energy_loss = F.mse_loss(pred_energy, target_energy)
        energy_loss = raw_energy_loss / target_energy_var

        # --- 3. Multi-Scale Spectral (MSS) Loss ---
        mss_loss = 0.0
        
        for n_fft in self.fft_sizes:
            hop_length = n_fft // 4
            window = torch.hann_window(n_fft).to(pred_audio.device)
            
            pred_stft = torch.stft(pred_audio.unsqueeze(0), n_fft=n_fft, hop_length=hop_length, 
                                   win_length=n_fft, window=window, return_complex=True)
            target_stft = torch.stft(target_audio.unsqueeze(0), n_fft=n_fft, hop_length=hop_length, 
                                     win_length=n_fft, window=window, return_complex=True)
            
            pred_mag = torch.abs(pred_stft) + 1e-7
            target_mag = torch.abs(target_stft) + 1e-7
            
            # Spectral Convergence (Linear magnitude differences)
            sc_loss = torch.norm(target_mag - pred_mag, p='fro') / (torch.norm(target_mag, p='fro') + 1e-7)
            
            # Log-Magnitude Loss (Captures the low-energy modal tails)
            log_loss = F.l1_loss(torch.log(pred_mag), torch.log(target_mag))
            
            mss_loss += (sc_loss + log_loss)
            
        # Average over the number of scales so the loss magnitude stays manageable
        mss_loss = mss_loss / len(self.fft_sizes)
 
        # --- 4. Low-Pass Loss ---
        N = pred_audio.shape[-1]
        freqs = torch.fft.rfftfreq(N, d=1.0 / self.sr).to(pred_audio.device)
        
        # Create a boolean mask for bins under the cutoff frequency
        valid_bins = freqs <= self.cutoff_hz
        
        # Compute full-signal FFT
        pred_fft = torch.fft.rfft(pred_audio)
        target_fft = torch.fft.rfft(target_audio)
        
        # Extract magnitude of only the low-frequency bins
        pred_mag_low = torch.abs(pred_fft)[valid_bins]
        target_mag_low = torch.abs(target_fft)[valid_bins]
        
        # L1 Loss on Log-Magnitude (Phase-blind, exactly like STFT but with 0.2Hz resolution)
        lowpass_loss = F.l1_loss(torch.log(pred_mag_low + 1e-7), torch.log(target_mag_low + 1e-7))

        # --- Final Compilation ---
        total_loss = (self.mse_weight * mse_loss) + \
                     (self.stft_weight * mss_loss) + \
                     (self.energy_weight * energy_loss) + \
                     (self.lowpass_weight * lowpass_loss)

        return total_loss