import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class Loss(nn.Module):

    def __init__(self, mse_weight=0.0, stft_weight=1.0,
                 energy_weight=0.0,
                 cutoff_hz=200.0, sr=44100,
                 fft_sizes=[64, 256, 1024, 4096],
                 n_mels=128):
        super().__init__()

        self.mse_weight = mse_weight
        self.stft_weight = stft_weight
        self.energy_weight = energy_weight
        self.sr = sr
        self.fft_sizes = fft_sizes
        self.n_mels = n_mels

        self.windows = nn.ParameterDict({
            str(n): nn.Parameter(torch.hann_window(n), requires_grad=False)
            for n in fft_sizes
        })
        
        # FIX WARNING: Scaliamo il numero di filtri Mel in base alla grandezza della FFT
        # (es: n=64 avrà 16 mels, n=256 avrà 64 mels, n=1024 e 4096 avranno 128 mels)
        self.mel_scales = nn.ModuleDict({
            str(n): torchaudio.transforms.MelScale(
                n_mels=min(self.n_mels, n // 4), 
                sample_rate=self.sr,
                n_stft=n // 2 + 1
            ) for n in fft_sizes
        })
            
        self.target_mel_cache = {}  # key: (device, n_fft), value: mel spectrogram tensor
        self.cached_target_audio = None

        self.eps = 1e-9

    def precompute_target_stft(self, target_audio):
        target_audio = target_audio.squeeze()
        device = target_audio.device
        
        self.cached_target_audio = target_audio
        self.target_mel_cache.clear()
        
        for n_fft in self.fft_sizes:
            hop = n_fft // 4
            window = self.windows[str(n_fft)].to(device)
            
            target_stft = torch.stft(target_audio, n_fft=n_fft, hop_length=hop,
                                     win_length=n_fft, window=window,
                                     return_complex=True, center=True)
            
            # Calcola la magnitudine
            target_mag = torch.abs(target_stft)
            
            # FIX ERRORE: Assicurati che il filtro Mel sia dello stesso device E dtype (float64) dell'audio
            mel_filter = self.mel_scales[str(n_fft)].to(device=device, dtype=target_mag.dtype)
            target_mel = mel_filter(target_mag)
            
            self.target_mel_cache[(device, n_fft)] = target_mel
    
    def forward(self, pred_audio, target_audio):
        peak = torch.max(torch.abs(target_audio)) + 1e-8
        norm_target = target_audio / peak
        norm_pred = pred_audio / peak
        pred_audio = pred_audio.squeeze()
        target_audio = target_audio.squeeze()

        device = pred_audio.device

        # =========================
        # 1. TIME-DOMAIN MSE
        # =========================
        mse_loss = F.mse_loss(norm_pred.squeeze(), norm_target.squeeze())

        # =========================
        # 2. ENERGY LOSS
        # =========================
        pred_energy = pred_audio**2
        target_energy = target_audio**2

        target_energy_var = torch.mean(target_energy**2).clamp_min(self.eps)
        energy_loss = F.mse_loss(pred_energy, target_energy) / target_energy_var

        # =========================
        # 3. MULTI-SCALE MEL-SPECTROGRAM
        # =========================
        mss_loss = torch.tensor(0.0, device=device, dtype=pred_audio.dtype)

        if self.stft_weight > 0:
            for n_fft in self.fft_sizes:
                hop = n_fft // 4
                window = self.windows[str(n_fft)].to(device)

                pred_stft = torch.stft(pred_audio, n_fft=n_fft, hop_length=hop,
                                       win_length=n_fft, window=window,
                                       return_complex=True, center=True)

                # Calcola la magnitudine del predetto
                pred_mag = torch.abs(pred_stft)
                
                # FIX ERRORE: Allinea il dtype anche durante il forward
                mel_filter = self.mel_scales[str(n_fft)].to(device=device, dtype=pred_mag.dtype)
                pred_mel = mel_filter(pred_mag)

                # Recupera il Mel target pre-calcolato
                target_mel = self.target_mel_cache[(device, n_fft)]

                # Clamp per evitare log(0) e divisioni per 0
                pred_mel = pred_mel.clamp_min(self.eps)
                target_mel = target_mel.clamp_min(self.eps)

                # Calcolo Loss sui Mel Spectrograms
                sc       = torch.norm(target_mel - pred_mel) / torch.norm(target_mel)
                log_loss = F.l1_loss(torch.log(pred_mel), torch.log(target_mel))

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