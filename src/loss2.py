import torch
import torch.nn as nn


class NMSELoss(nn.Module):
    """
    Normalised Mean Square Error:

        NMSE = ||pred - target||² / ||target||²

    Scale-invariant: a global amplitude offset in both signals cancels out,
    so the loss only measures waveform shape differences, not absolute level.
    """

    def __init__(self):
        super().__init__()
        self.eps = 1e-9

    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        pred   = pred_audio.squeeze()
        target = target_audio.squeeze()

        num = torch.sum((pred - target) ** 2)
        den = torch.sum(target ** 2).clamp(min=self.eps)

        return num / den
