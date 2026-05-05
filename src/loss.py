import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):

    def __init__(self, mse_weight=0.0, stft_weight=1.0,
                 energy_weight=1.0, lowpass_weight=10.0,
                 cutoff_hz=200.0, sr=44100,
                 fft_sizes=[64, 256, 1024, 4096]):
        super().__init__()

        self.mse_weight = mse_weight
        self.stft_weight = stft_weight
        self.energy_weight = energy_weight
        self.lowpass_weight = lowpass_weight

        self.cutoff_hz = cutoff_hz
        self.sr = sr
        self.fft_sizes = fft_sizes

        # cache finestre (IMPORTANTISSIMO)
        self.windows = nn.ParameterDict({
            str(n): nn.Parameter(torch.hann_window(n), requires_grad=False)
            for n in fft_sizes
        })

        self.eps = 1e-7

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
        # 4. LOWPASS LOSS (SMOOTH)
        # =========================
        N = pred_audio.shape[-1]

        freqs = torch.fft.rfftfreq(N, d=1.0/self.sr).to(device)

        # smooth mask (differenziabile)
        cutoff = torch.tensor(self.cutoff_hz, device=device)
        mask = torch.sigmoid((cutoff - freqs) / 5.0)

        pred_fft = torch.fft.rfft(pred_audio)
        target_fft = torch.fft.rfft(target_audio)

        pred_mag = torch.abs(pred_fft).clamp_min(self.eps)
        target_mag = torch.abs(target_fft).clamp_min(self.eps)

        lowpass_loss = F.l1_loss(
            torch.log(pred_mag) * mask,
            torch.log(target_mag) * mask
        )

        # =========================
        # FINAL
        # =========================
        total_loss = (
            self.mse_weight * mse_loss +
            self.stft_weight * mss_loss +
            self.energy_weight * energy_loss +
            self.lowpass_weight * lowpass_loss
        )

        return total_loss