import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleSpectralLoss(nn.Module):
    """
    Computes the multi-scale spectrogram loss between the generated IR and target IR.
    Combines linear and log-magnitude STFT differences to capture both transients and decays.
    """
    def __init__(self, fft_sizes=(128, 256, 512, 1024, 2048, 4096)):
        super(MultiScaleSpectralLoss, self).__init__()
        self.fft_sizes = fft_sizes

    def compute_spectrogram(self, x: torch.Tensor, n_fft: int) -> torch.Tensor:
        # Ensure input has a batch dimension for STFT processing
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        hop_length = n_fft // 4
        win_length = n_fft
        window = torch.hann_window(win_length).to(x.device)
        
        # Calculate STFT
        stft_out = torch.stft(
            x, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            win_length=win_length, 
            window=window, 
            return_complex=True
        )
        
        # Return magnitude bounded by a small epsilon to prevent log(0)
        return torch.abs(stft_out) + 1e-7

    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        
        for n_fft in self.fft_sizes:
            S_pred = self.compute_spectrogram(pred_audio, n_fft)
            S_target = self.compute_spectrogram(target_audio, n_fft)
            
            # Linear loss: focuses on initial high-energy transient peaks
            loss_lin = F.l1_loss(S_pred, S_target)
            
            # Log loss: focuses on quiet, long-tail mode decays
            loss_log = F.l1_loss(torch.log(S_pred), torch.log(S_target))
            
            total_loss += (loss_lin + loss_log)
            
        return total_loss / len(self.fft_sizes)