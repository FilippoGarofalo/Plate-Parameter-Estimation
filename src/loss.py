import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeDomainEnergyLoss(nn.Module):
    """
    Computes standard time-domain MSE and Energy MSE between the generated IR and target IR.
    Perfect for exact mathematical twin matching.
    """
    def __init__(self, energy_weight: float = 1.0):
        super(TimeDomainEnergyLoss, self).__init__()
        # Weight to balance the raw waveform loss vs the energy power loss
        self.energy_weight = energy_weight

    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        """
        Calculates the combined MSE and Energy loss.
        """
        # Ensure identical shapes (1D tensors)
        pred_audio = pred_audio.squeeze()
        target_audio = target_audio.squeeze()
        
        # ---------------------------------------------------------
        # 1. Standard Time-Domain MSE
        # ---------------------------------------------------------
        # This penalizes exact sample-by-sample differences. 
        # It forces the optimizer to perfectly match phase, frequency, and amplitude.
        mse_loss = F.mse_loss(pred_audio, target_audio)
        
        # ---------------------------------------------------------
        # 2. Instantaneous Energy MSE
        # ---------------------------------------------------------
        # Squaring the audio gives us the energy/power of the signal.
        # This heavily penalizes mismatched transient spikes and ensures
        # the overall exponential decay rate (T60) matches perfectly.
        pred_energy = pred_audio ** 2
        target_energy = target_audio ** 2
        
        energy_loss = F.mse_loss(pred_energy, target_energy)
        
        # ---------------------------------------------------------
        # 3. Total Combined Loss
        # ---------------------------------------------------------
        total_loss = mse_loss + (self.energy_weight * energy_loss)
        
        return total_loss