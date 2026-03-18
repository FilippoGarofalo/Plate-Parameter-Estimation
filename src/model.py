import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DifferentiableModalPlate(nn.Module):
    def __init__(self, sample_rate: int = 44100):
        super(DifferentiableModalPlate, self).__init__()
        self.sample_rate = sample_rate
        self.k = 1.0 / sample_rate
        self.fmax = 10000.0
        self.maxOm = self.fmax * 2 * np.pi
        
     
        # 1. FIXED PARAMETERS
        self.register_buffer('Lx', torch.tensor(1.0))
        self.register_buffer('tau_0', torch.tensor(6.0))
        self.register_buffer('tau_1', torch.tensor(2.0))
        self.register_buffer('loss_f1', torch.tensor(500.0))
        
        # Rayleigh damping constants
        OmDamp1 = 0.0
        OmDamp2 = 2 * np.pi * self.loss_f1
        dOmSq = OmDamp2**2 - OmDamp1**2
        
        alpha = 3 * np.log(10) / dOmSq * (OmDamp2**2 / self.tau_0 - OmDamp1**2 / self.tau_1)
        beta = 3 * np.log(10) / dOmSq * (1 / self.tau_1 - 1 / self.tau_0)
        self.register_buffer('alpha', torch.tensor(alpha))
        self.register_buffer('beta', torch.tensor(beta))

        # Fixed modal grid to prevent graph breaking (Max modes up to 10kHz)
        M_max, N_max = 80, 80 
        m_idx = torch.arange(1, M_max + 1)
        n_idx = torch.arange(1, N_max + 1)
        grid_m, grid_n = torch.meshgrid(m_idx, n_idx, indexing='ij')
        self.register_buffer('m_vec', grid_m.flatten().float())
        self.register_buffer('n_vec', grid_n.flatten().float())

        
        # 2. LEARNABLE PARAMETERS
        self.mu_raw = nn.Parameter(torch.tensor(0.0))
        self.D_over_mu_raw = nn.Parameter(torch.tensor(0.0))
        self.T0_over_mu_raw = nn.Parameter(torch.tensor(0.0))
        self.Ly_raw = nn.Parameter(torch.tensor(0.0))
        self.xo_raw = nn.Parameter(torch.tensor(0.0))
        self.yo_raw = nn.Parameter(torch.tensor(0.0))

    def get_physical_parameters(self):
        """
        Applies mathematical bounds through differentiable tranformations
        """
        # Softplus for positivity constraint
        mu = F.softplus(self.mu_raw) + 1e-4
        D_over_mu = F.softplus(self.D_over_mu_raw) + 1e-4
        T0_over_mu = F.softplus(self.T0_over_mu_raw) + 1e-4
        
        # Sigmoid maps to bounded ranges (given by the challenge specs)
        Ly = 1.1 + (4.0 - 1.1) * torch.sigmoid(self.Ly_raw)
        xo = (0.49 * self.Lx) + ((1.0 - 0.49) * self.Lx) * torch.sigmoid(self.xo_raw)
        yo = (0.51 * Ly) + ((1.0 - 0.51) * Ly) * torch.sigmoid(self.yo_raw)
        
        return mu, D_over_mu, T0_over_mu, Ly, xo, yo

    def forward(self, num_samples: int) -> torch.Tensor:
        #Retrieve scaled physical parameters
        mu, D_over_mu, T0_over_mu, Ly, xo, yo = self.get_physical_parameters()

        #Initialize input position 
        xi = 0.335 * self.Lx
        yi = 0.467 * Ly

        # ---------------------------------------------------------
        # A. FREQUENCIES & MASKS 
        # ---------------------------------------------------------

        # (from wave numbers)
        g1 = (self.m_vec * np.pi / self.Lx)**2 + (self.n_vec * np.pi / Ly)**2
        g2 = g1 * g1
        
        #Dispersion relation
        omega_sq = T0_over_mu * g1 + D_over_mu * g2
        omega = torch.sqrt(torch.relu(omega_sq)) 
        
        #"Differential gate" for frequencies >10kHz and <20Hz
        temp = 100.0 # Temperature scaling for sigmoid steepness
        mask_high = torch.sigmoid((self.maxOm - omega) / temp)
        mask_low = torch.sigmoid((omega - (20 * 2 * np.pi)) / temp)
        valid_modes_mask = mask_high * mask_low

        # ---------------------------------------------------------
        # B. AMPLITUDES & DECAYS
        # ---------------------------------------------------------
        InWeight = torch.cos(xi * np.pi * self.m_vec / self.Lx) * torch.cos(yi * np.pi * self.n_vec / Ly)
        OutWeight = torch.cos(xo * np.pi * self.m_vec / self.Lx) * torch.cos(yo * np.pi * self.n_vec / Ly)
        
        sigma = self.alpha + self.beta * omega**2
        ms = 0.25 * mu * self.Lx * Ly 
        
        # Apply validity mask to the initial amplitudes
        P = (OutWeight * InWeight * self.k**2 * torch.exp(-sigma * self.k) / ms) * valid_modes_mask
        
        return ir_out