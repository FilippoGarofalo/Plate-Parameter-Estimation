
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):

    def __init__(self, mse_weight=0.0, stft_weight=1.0,
                 energy_weight=0.0,
                 cutoff_hz=200.0, sr=44100,
                 fft_sizes=[64, 256, 1024, 4096]):
        super().__init__()

        
        self.mse_weight = mse_weight              
        self.stft_weight = stft_weight            
        self.energy_weight = energy_weight       
        
        self.sr = sr                            
        self.fft_sizes = fft_sizes                
        self.cutoff_hz = cutoff_hz                


        self.windows = nn.ParameterDict({
            str(n): nn.Parameter(torch.hann_window(n), requires_grad=False)
            for n in fft_sizes
        })
        
        
        self.target_stft_cache = {}               
        self.cached_target_audio = None

        self.eps = 1e-9                           

    def precompute_target_stft(self, target_audio):
        
        target_audio = target_audio.squeeze()
        device = target_audio.device
        
        self.cached_target_audio = target_audio
        self.target_stft_cache.clear()
        
        for n_fft in self.fft_sizes:
            hop = n_fft // 4
            window = self.windows[str(n_fft)].to(device)
            
            
            target_stft = torch.stft(target_audio, n_fft=n_fft, hop_length=hop,
                                     win_length=n_fft, window=window,
                                     return_complex=True, center=True)
            
            self.target_stft_cache[(device, n_fft)] = target_stft
    
    def forward(self, pred_audio, target_audio):

        pred_audio = pred_audio.squeeze()
        target_audio = target_audio.squeeze()

        device = pred_audio.device

        if torch.isnan(pred_audio).any() or torch.isinf(pred_audio).any():
            print(f"WARNING: pred_audio contiene NaN/inf! Returning large loss.")
            return torch.tensor(1e6, device=device, dtype=pred_audio.dtype)

        
        pred_energy = pred_audio ** 2
        target_energy = target_audio ** 2

        target_energy_var = torch.mean(target_energy ** 2)
        target_energy_var = torch.clamp(target_energy_var, min=self.eps)
        
        energy_loss = F.mse_loss(pred_energy, target_energy) / target_energy_var
        
        mss_loss = torch.tensor(0.0, device=device, dtype=pred_audio.dtype)

        for n_fft in self.fft_sizes:
            hop = n_fft // 4  
            window = self.windows[str(n_fft)].to(device)

            
            pred_stft = torch.stft(pred_audio, n_fft=n_fft, hop_length=hop,
                                   win_length=n_fft, window=window,
                                   return_complex=True, center=True)
            
            target_stft = self.target_stft_cache[(device, n_fft)]

            
            pred_mag = torch.abs(pred_stft)      
            target_mag = torch.abs(target_stft)  

            
            pred_mag_clipped = torch.clamp(pred_mag, min=self.eps)
            target_mag_clipped = torch.clamp(target_mag, min=self.eps)
            
            mag_diff = target_mag - pred_mag  
            
            sc_numerator = torch.norm(mag_diff, p='fro')
            sc_denominator = torch.norm(target_mag, p='fro')
            sc_denominator = torch.clamp(sc_denominator, min=self.eps)
            
            sc_loss = sc_numerator / sc_denominator
            
            log_target = torch.log(target_mag_clipped)
            log_pred = torch.log(pred_mag_clipped)
            
            lm_loss = F.l1_loss(log_pred, log_target, reduction='mean')
            
            scale_loss = sc_loss + lm_loss
            mss_loss = mss_loss + scale_loss

        mss_loss = mss_loss / len(self.fft_sizes)
        
        total_loss = (
            self.stft_weight * mss_loss +
            self.energy_weight * energy_loss
        )
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"WARNING: Loss è NaN/inf!")
            print(f"   mss_loss={mss_loss.item():.6e}, energy_loss={energy_loss.item():.6e}")
            return torch.tensor(1e6, device=device, dtype=pred_audio.dtype)

        return total_loss


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_audio, target_audio):
        return F.mse_loss(pred_audio, target_audio)