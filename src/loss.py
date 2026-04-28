import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeDomainEnergyLoss(nn.Module):
    """
    Computes a combined Relative Time-Domain MSE and Log-Magnitude STFT Loss.
    """
    def __init__(self, mse_weight: float = 1.0, stft_weight: float = 1.0, energy_weight: float = 1.0):
        super(TimeDomainEnergyLoss, self).__init__()
        self.mse_weight = mse_weight
        self.stft_weight = stft_weight
        self.energy_weight = energy_weight
        #self.lowpass_weight = lowpass_weight

        # STFT parameters
        self.n_fft = 2048
        self.hop_length = 512
        self.win_length = 2048

        # Low-pass filter via FFT: pre-compute the mask
        #self.cutoff_hz = cutoff_hz
        #self.sr = sr

    def _lowpass(self, x):
        N = x.shape[-1]
        X = torch.fft.rfft(x)
        freqs = torch.fft.rfftfreq(N, d=1.0 / self.sr).to(x.device)
        mask = (freqs <= self.cutoff_hz).float()
        return torch.fft.irfft(X * mask, n=N)

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
 
        total_loss = (self.mse_weight * mse_loss) + (self.stft_weight * stft_loss) + (self.energy_weight * energy_loss)

        # Low-pass MSE (guides T0/mu)
        # pred_low = self._lowpass(pred_audio)
        # target_low = self._lowpass(target_audio)
        #lowpass_loss = F.mse_loss(pred_low, target_low)

        #total_loss = total_loss + self.lowpass_weight * lowpass_loss

        return total_loss