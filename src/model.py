import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DifferentiableModalPlate(nn.Module):

    def __init__(self, sample_rate: int = 44100, plate_params: dict = None,
                 dtype: torch.dtype = torch.float64):
        super(DifferentiableModalPlate, self).__init__()
        self.sample_rate = sample_rate
        self.k = 1.0 / sample_rate
        self.fmax = 10000.0
        self.maxOm = self.fmax * 2 * np.pi
        self.dtype = dtype

        # 1. FIXED PARAMETERS
        self.register_buffer('Lx', torch.tensor(0.50, dtype=dtype))
        self.register_buffer('tau_0', torch.tensor(6.0, dtype=dtype))
        self.register_buffer('tau_1', torch.tensor(1.0, dtype=dtype))
        self.register_buffer('loss_f1', torch.tensor(500.0, dtype=dtype))

        # Rayleigh damping constants
        OmDamp1 = 0.0
        OmDamp2 = 2 * np.pi * self.loss_f1
        dOmSq = OmDamp2**2 - OmDamp1**2

        alpha = 3 * np.log(10) / dOmSq * (OmDamp2**2 / self.tau_0 - OmDamp1**2 / self.tau_1)
        beta = 3 * np.log(10) / dOmSq * (1 / self.tau_1 - 1 / self.tau_0)
        self.register_buffer('alpha', alpha.clone().detach().to(dtype))
        self.register_buffer('beta', beta.clone().detach().to(dtype))

        # Fixed modal grid to prevent graph breaking (Max modes up to 10kHz)
        M_max, N_max = 80, 80
        m_idx = torch.arange(1, M_max + 1)
        n_idx = torch.arange(1, N_max + 1)
        grid_m, grid_n = torch.meshgrid(m_idx, n_idx, indexing='ij')
        self.register_buffer('m_vec', grid_m.flatten().clone().detach().to(dtype))
        self.register_buffer('n_vec', grid_n.flatten().clone().detach().to(dtype))

        # 2. LEARNABLE PARAMETERS
        if plate_params is None:
            self.mu_raw = nn.Parameter(torch.tensor(0.0, dtype=dtype))
            self.D_over_mu_raw = nn.Parameter(torch.tensor(0.0, dtype=dtype))
            self.T0_over_mu_raw = nn.Parameter(torch.tensor(0.0, dtype=dtype))
            self.Ly_raw = nn.Parameter(torch.tensor(0.0, dtype=dtype))
            self.xo_raw = nn.Parameter(torch.tensor(0.0, dtype=dtype))
            self.yo_raw = nn.Parameter(torch.tensor(0.0, dtype=dtype))
        else:
            print("Initializing with provided plate parameters...")
            self.mu_raw = nn.Parameter(torch.tensor(plate_params['mu_raw'], dtype=dtype))
            self.D_over_mu_raw = nn.Parameter(torch.tensor(plate_params['D_over_mu_raw'], dtype=dtype))
            self.T0_over_mu_raw = nn.Parameter(torch.tensor(plate_params['T0_over_mu_raw'], dtype=dtype))
            self.Ly_raw = nn.Parameter(torch.tensor(plate_params['Ly_raw'], dtype=dtype))
            self.xo_raw = nn.Parameter(torch.tensor(plate_params['xo_raw'], dtype=dtype))
            self.yo_raw = nn.Parameter(torch.tensor(plate_params['yo_raw'], dtype=dtype))

    def get_physical_parameters(self):
        """
        Applies mathematical bounds through differentiable tranformations
        """
        mu = F.softplus(self.mu_raw) + 1e-4
        D_over_mu = F.softplus(self.D_over_mu_raw) + 1e-4
        T0_over_mu = F.softplus(self.T0_over_mu_raw) + 1e-4
        
        # Sigmoid maps to bounded ranges (given by the challenge specs)
        #Ly = 1.1 + (4.0 - 1.1) * torch.sigmoid(self.Ly_raw)
        #xo = (0.49 * self.Lx) + ((1.0 - 0.49) * self.Lx) * torch.sigmoid(self.xo_raw)
        #yo = (0.51 * Ly) + ((1.0 - 0.51) * Ly) * torch.sigmoid(self.yo_raw)

        # tanh maps to bounded ranges with steeper gradients near the boundaries, which can help optimization
        Ly = 1.1 + (4.0 - 1.1) * ((torch.tanh(self.Ly_raw) + 1.0) / 2.0)
        xo = (0.49 * self.Lx) + ((1.0 - 0.49) * self.Lx) * ((torch.tanh(self.xo_raw) + 1.0) / 2.0)
        yo = (0.51 * Ly) + ((1.0 - 0.51) * Ly) * ((torch.tanh(self.yo_raw) + 1.0) / 2.0)
        
        return mu, D_over_mu, T0_over_mu, Ly, xo, yo

    def forward(self, duration: float = 1.0, normalize: bool = True, velCalc: bool = False) -> torch.Tensor:
        """
        Synthesize impulse response using modal synthesis with full differentiability.
        
        Args:
            duration: Duration in seconds (default: 1.0)
            normalize: Normalize output by peak amplitude (default: True)
            velCalc: If True, compute velocity output; if False, displacement (default: False)
            
        Returns:
            torch.Tensor: Synthesized impulse response waveform (displacement or velocity)
        """
        #Retrieve scaled physical parameters
        mu, D_over_mu, T0_over_mu, Ly, xo, yo = self.get_physical_parameters()

        #Initialize input position 
        xi = 0.10 * self.Lx
        yi = 0.10 * Ly

        # A. FREQUENCIES & MASKS 
        # (from wave numbers)
        #kˆ2 = (m*pi/Lx)² + (n*pi/Ly)²
        g1 = (self.m_vec * np.pi / self.Lx)**2 + (self.n_vec * np.pi / Ly)**2
        #kˆ4 = kˆ2² Biharmonic operator
        g2 = g1 * g1
        
        # Dispersion relation
        omega_sq = T0_over_mu * g1 + D_over_mu * g2
        omega = torch.sqrt(torch.relu(omega_sq))
        
        #hard-coded masking
        mask_high = (omega <= self.maxOm).to(dtype=omega.dtype)   # era .float() → float32
        mask_low  = (omega >= (20 * 2 * np.pi)).to(dtype=omega.dtype)
        valid_modes_mask = mask_high * mask_low

        # B. AMPLITUDES & DECAYS

        #spatial filtering weights for input and output positions
        InWeight = torch.cos(xi * np.pi * self.m_vec / self.Lx) * torch.cos(yi * np.pi * self.n_vec / Ly)
        OutWeight = torch.cos(xo * np.pi * self.m_vec / self.Lx) * torch.cos(yo * np.pi * self.n_vec / Ly)
        
        sigma = self.alpha + self.beta * omega**2
        ms = 0.25 * mu * self.Lx * Ly 
        
        # Apply validity mask to the initial amplitudes
        P = (OutWeight * InWeight * self.k**2 * torch.exp(-sigma * self.k) / ms) * valid_modes_mask

        # C. VECTORIZED MODAL SYNTHESIS        
        num_samples = int(self.sample_rate * duration)
        n_vec = torch.arange(num_samples, device=P.device, dtype=self.dtype)
        
        # Broadcasting dimensions
        omega_col = omega.unsqueeze(1)
        sigma_col = sigma.unsqueeze(1)
        P_col = P.unsqueeze(1)
        n_row = n_vec.unsqueeze(0)
        
        # 1. Calculate continuous displacement 
        # Entrambi gli indici shiftati di 1
        decay_env = torch.exp(-sigma_col * (n_row - 1) * self.k)  # era n_row
        sine_num  = torch.sin(n_row * omega_col * self.k)          # era n_row + 1
        sine_den  = torch.sin(omega_col * self.k) + 1e-8
        
        mode_waveforms = P_col * decay_env * (sine_num / sine_den)
        
        # 2. Sum all modes to create the 1D displacement signal
        displacement_out = torch.sum(mode_waveforms, dim=0)
        
        # 3. Handle Velocity vs Displacement 
        if velCalc:
            y_prev_tensor = torch.cat([torch.tensor([0.0], device=P.device, dtype=self.dtype), displacement_out[:-1]])
            ir_out = (displacement_out - y_prev_tensor) / self.k
        else:
            ir_out = displacement_out

        # 4. Normalization
        if normalize:
            peak = torch.max(torch.abs(ir_out)) + 1e-8
            ir_out = ir_out / peak
        
        return ir_out