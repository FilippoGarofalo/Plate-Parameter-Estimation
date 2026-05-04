import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeDomainEnergyLoss(nn.Module):
    """
    Computes a combined Relative Time-Domain MSE and Log-Magnitude STFT Loss.
    """
    def __init__(self, mse_weight: float = 0.0, stft_weight: float = 1.0, 
                 energy_weight: float = 1.0, lowpass_weight: float = 10.0, 
                 cutoff_hz: float = 200.0, sr: float = 44100):
        super(TimeDomainEnergyLoss, self).__init__()
        self.mse_weight = mse_weight
        self.stft_weight = stft_weight
        self.energy_weight = energy_weight
        self.lowpass_weight = lowpass_weight
        self.cutoff_hz = cutoff_hz
        self.sr = sr

        # STFT parameters
        self.n_fft = 2048
        self.hop_length = 512
        self.win_length = 2048

        


    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        pred_audio = pred_audio.squeeze()
        target_audio = target_audio.squeeze()
    
        target_variance = torch.mean(target_audio ** 2) + 1e-8
        raw_mse = F.mse_loss(pred_audio, target_audio)
        mse_loss = raw_mse / target_variance
        
        pred_energy = pred_audio ** 2
        target_energy = target_audio ** 2
        
        target_energy_var = torch.mean(target_energy ** 2) + 1e-8
        raw_energy_loss = F.mse_loss(pred_energy, target_energy)
        energy_loss = raw_energy_loss / target_energy_var

        window = torch.hann_window(self.win_length).to(pred_audio.device)
        
        pred_stft = torch.stft(pred_audio.unsqueeze(0), n_fft=self.n_fft, hop_length=self.hop_length, 
                               win_length=self.win_length, window=window, return_complex=True)
        target_stft = torch.stft(target_audio.unsqueeze(0), n_fft=self.n_fft, hop_length=self.hop_length, 
                                 win_length=self.win_length, window=window, return_complex=True)
        
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        
        stft_loss = F.l1_loss(torch.log(pred_mag + 1e-7), torch.log(target_mag + 1e-7))
 
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
                     (self.stft_weight * stft_loss) + \
                     (self.energy_weight * energy_loss) + \
                     (self.lowpass_weight * lowpass_loss)

        return total_loss

        return total_loss