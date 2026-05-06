import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        pred = pred_audio.squeeze()
        target = target_audio.squeeze()
        return F.mse_loss(pred, target)
