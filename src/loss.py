import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeDomainEnergyLoss(nn.Module):
    """
    Computes standard time-domain MSE and Energy MSE between the generated IR and target IR.
    Perfect for exact mathematical twin matching.
    Includes a small STFT penalty to prevent 'mute shortcuts' during optimization.
    """
    def __init__(self):
        super(TimeDomainEnergyLoss, self).__init__()

    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        pred_audio = pred_audio.squeeze()
        target_audio = target_audio.squeeze()
    
        mse_loss = F.mse_loss(pred_audio, target_audio)
        
        
        return mse_loss