import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):

    def __init__(self, mse_weight=0.0, stft_weight=1.0,
                 cutoff_hz=200.0, sr=44100,
                 fft_sizes=[64, 256, 1024, 4096]):
        super().__init__()

        self.mse_weight = mse_weight
        self.stft_weight = stft_weight

        self.cutoff_hz = cutoff_hz
        self.sr = sr
        self.fft_sizes = fft_sizes

        self.windows = nn.ParameterDict({
            str(n): nn.Parameter(torch.hann_window(n), requires_grad=False)
            for n in fft_sizes
        })

        self.eps = 1e-4

    def forward(self, pred_audio, target_audio):

        pred_audio = pred_audio.squeeze()
        target_audio = target_audio.squeeze()

        device = pred_audio.device

        # =========================
        # 1. MSE NORMALIZED
        # =========================
        target_var = torch.mean(target_audio**2).clamp_min(self.eps)
        mse_loss = F.mse_loss(pred_audio, target_audio) / target_var

        # =========================
        # 2. ENERGY LOSS
        # =========================
        pred_energy = pred_audio**2
        target_energy = target_audio**2

        target_energy_var = torch.mean(target_energy**2).clamp_min(self.eps)
        energy_loss = F.mse_loss(pred_energy, target_energy) / target_energy_var

        # =========================
        # 3. MULTI-SCALE STFT
        # =========================
        mss_loss = 0.0

        for n_fft in self.fft_sizes:
            hop = n_fft // 4
            window = self.windows[str(n_fft)].to(device)

            pred_stft = torch.stft(pred_audio, n_fft=n_fft, hop_length=hop,
                                   win_length=n_fft, window=window,
                                   return_complex=True, center=True)

            target_stft = torch.stft(target_audio, n_fft=n_fft, hop_length=hop,
                                     win_length=n_fft, window=window,
                                     return_complex=True, center=True)

            pred_mag = torch.abs(pred_stft).clamp_min(self.eps)
            target_mag = torch.abs(target_stft).clamp_min(self.eps)

            # spectral convergence (più veloce)
            sc = torch.norm(target_mag - pred_mag) / torch.norm(target_mag)

            # log-magnitude
            log_loss = F.l1_loss(torch.log(pred_mag), torch.log(target_mag))

            mss_loss = mss_loss + (sc + log_loss)

        mss_loss = mss_loss / len(self.fft_sizes)

        # =========================
        # FINAL
        # =========================
        total_loss = (
            self.mse_weight * mse_loss +
            self.stft_weight * mss_loss +
            self.energy_weight * energy_loss
        )

        return total_loss