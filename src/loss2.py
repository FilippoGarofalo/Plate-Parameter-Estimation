import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.eps = 1e-9

    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        pred = pred_audio.squeeze()
        target = target_audio.squeeze()
        
        # Normalize by target energy (RMS)
        target_energy = torch.sqrt(torch.mean(target**2) + self.eps)
        pred_norm = pred / target_energy
        target_norm = target / target_energy

        # Simply compare RMS-normalised signals — no second division by energy^2
        return F.mse_loss(pred_norm, target_norm)
