import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeDomainEnergyLoss(nn.Module):
    """
    Computes standard time-domain MSE and Energy MSE between the generated IR and target IR.
    Perfect for exact mathematical twin matching.
    Includes a small STFT penalty to prevent 'mute shortcuts' during optimization.
    """
    def __init__(self, energy_weight: float = 1.0, stft_weight: float = 0.1):
        super(TimeDomainEnergyLoss, self).__init__()
        self.energy_weight = energy_weight
        self.stft_weight = stft_weight
        
        self.n_fft = 2048
        self.hop_length = 512
        self.win_length = 2048

    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        """
        Calculates the combined MSE, Energy loss, and STFT penalty.
        """
        pred_audio = pred_audio.squeeze()
        target_audio = target_audio.squeeze()
    
        mse_loss = F.mse_loss(pred_audio, target_audio)
        
        pred_energy = pred_audio ** 2
        target_energy = target_audio ** 2
        
        energy_loss = F.mse_loss(pred_energy, target_energy)
        
        window = torch.hann_window(self.win_length).to(pred_audio.device)
        
        pred_stft = torch.stft(pred_audio.unsqueeze(0), n_fft=self.n_fft, hop_length=self.hop_length, 
                               win_length=self.win_length, window=window, return_complex=True)
        target_stft = torch.stft(target_audio.unsqueeze(0), n_fft=self.n_fft, hop_length=self.hop_length, 
                                 win_length=self.win_length, window=window, return_complex=True)
        
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        
        stft_loss = F.l1_loss(torch.log(pred_mag + 1e-7), torch.log(target_mag + 1e-7))
 
        total_loss = mse_loss + (self.energy_weight * energy_loss) + (self.stft_weight * stft_loss)
        
        return total_loss