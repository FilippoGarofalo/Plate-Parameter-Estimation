import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleSpectralLoss(nn.Module):
    """
    Computes the multi-scale spectrogram loss between the generated IR and target IR.
    Computes both linear and log-magnitude STFT differences
    """
    def __init__(self, fft_sizes=(32, 64, 128, 256, 512, 1024, 2048, 4096)):
        super(MultiScaleSpectralLoss, self).__init__()
        self.fft_sizes = fft_sizes

    def compute_spectrogram(self, x: torch.Tensor, n_fft: int) -> torch.Tensor:
        """
        Computes the magnitude spectrogram of a 1D audio signal.
        """
        # Changes shape from [num_samples] to [1, num_samples]
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        hop_length = n_fft // 4
        win_length = n_fft
        
        # Create a Hann window 
        window = torch.hann_window(win_length, device=x.device)
        
        # Calculate the Short-Time Fourier Transform
        # return_complex=True is the modern PyTorch standard for stft
        stft_out = torch.stft(
            x, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            win_length=win_length, 
            window=window, 
            return_complex=True
        )
        
        # Add a small epsilon (1e-7) to prevent log(0).
        magnitude = torch.abs(stft_out) + 1e-7
        return magnitude

    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        """
        Calculates the accumulated loss across all FFT scales.
        """
        total_loss = 0.0
        
        for n_fft in self.fft_sizes:
            # Compute spectrograms
            S_pred = self.compute_spectrogram(pred_audio, n_fft)
            S_target = self.compute_spectrogram(target_audio, n_fft)
            
            # Linear Scale Loss (L1 distance)
            loss_lin = F.l1_loss(S_pred, S_target)
            
            # Log Scale Loss (L1 distance of logarithms)
            loss_log = F.l1_loss(torch.log(S_pred), torch.log(S_target))
            
            total_loss += (loss_lin + loss_log)
            
        # Return the average loss across all different FFT sizes 
        return total_loss / len(self.fft_sizes)