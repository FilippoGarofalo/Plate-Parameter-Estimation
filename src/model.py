import torch
from torch.utils.checkpoint import checkpoint
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DifferentiableModalPlate(nn.Module):

    def __init__(self, sample_rate: int = 44100, plate_params: dict = None,
                 dtype: torch.dtype = torch.float64):
        super(DifferentiableModalPlate, self).__init__()
        self.sample_rate = sample_rate
        self.k = 1.0 / sample_rate
        self.fmax = 20000.0
        self.maxOm = self.fmax * 2 * np.pi
        self.dtype = dtype

        # 1. FIXED PARAMETERS
        self.register_buffer('Lx', torch.tensor(1.0, dtype=dtype))
        self.register_buffer('tau_0', torch.tensor(6.0, dtype=dtype))
        self.register_buffer('tau_1', torch.tensor(2.0, dtype=dtype))
        self.register_buffer('loss_f1', torch.tensor(500.0, dtype=dtype))

        self.register_buffer('mu_scale', torch.tensor(110.0, dtype=dtype))
        self.register_buffer('D_over_mu_scale', torch.tensor(1100.0, dtype=dtype))
        self.register_buffer('T0_over_mu_scale', torch.tensor(500.0, dtype=dtype))

        # Rayleigh damping constants
        OmDamp1 = 0.0
        OmDamp2 = 2 * np.pi * self.loss_f1
        dOmSq = OmDamp2**2 - OmDamp1**2

        alpha = 3 * np.log(10) / dOmSq * (OmDamp2**2 / self.tau_0 - OmDamp1**2 / self.tau_1)
        beta = 3 * np.log(10) / dOmSq * (1 / self.tau_1 - 1 / self.tau_0)
        self.register_buffer('alpha', alpha.clone().detach().to(dtype))
        self.register_buffer('beta', beta.clone().detach().to(dtype))

        # Fixed modal grid to prevent graph breaking (Max modes up to 10kHz)
        M_max, N_max = 120, 120
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
        def map_range_linear(x, min_v, max_v, scale=1.0):
            return min_v + (max_v - min_v) * ((torch.tanh(x * scale) + 1.0) / 2.0)

        def map_range_log(x, min_v, max_v, scale=1.0):
            norm_x = (torch.tanh(x * scale) + 1.0) / 2.0
            log_min = np.log10(min_v)
            log_max = np.log10(max_v)
            log_val = log_min + (log_max - log_min) * norm_x
            return 10.0 ** log_val

        mu = map_range_log(self.mu_raw, 2.43, 106.15)
        
        # D/mu dominates high frequencies, so we slow it down to stop it from taking over
        D_over_mu = map_range_log(self.D_over_mu_raw, 0.05, 1005.9, scale=0.5)
        
        # T0/mu has tiny gradients, so we multiply its speed by 10
        T0_over_mu = map_range_log(self.T0_over_mu_raw, 0.0001, 411.52, scale=10.0)

        Ly = map_range_linear(self.Ly_raw, 1.1, 4.0)
        xo = map_range_linear(self.xo_raw, 0.51 * self.Lx, 1.0 * self.Lx)
        yo = map_range_linear(self.yo_raw, 0.51 * Ly, 1.0 * Ly)
        
        return mu, D_over_mu, T0_over_mu, Ly, xo, yo
    
    def compute_chunk(self, w, s, p, n):
            decay = torch.exp(-s * (n - 1) * self.k)
            sin_n = torch.sin(n * w * self.k)
            sin_d = torch.sin(w * self.k) + 1e-8
            mode_waves = p * decay * (sin_n / sin_d)
            return torch.sum(mode_waves, dim=0)
    
    def forward(self, duration: float = 1.0, normalize: bool = True, velCalc: bool = False) -> torch.Tensor:
        mu, D_over_mu, T0_over_mu, Ly, xo, yo = self.get_physical_parameters()

        frac_xi = 0.335  
        frac_yi = 0.467  
        frac_xo = xo / self.Lx 
        frac_yo = yo / Ly

        # A. FREQUENCIES & MASKS 
        g1 = (self.m_vec * np.pi / self.Lx)**2 + (self.n_vec * np.pi / Ly)**2
        g2 = g1 * g1
        
        omega_sq = T0_over_mu * g1 + D_over_mu * g2
        omega = torch.sqrt(torch.relu(omega_sq))
        
        # B. AMPLITUDES & DECAYS 
        InWeight = torch.cos(frac_xi * np.pi * self.m_vec) * torch.cos(frac_yi * np.pi * self.n_vec)
        OutWeight = torch.cos(frac_xo * np.pi * self.m_vec) * torch.cos(frac_yo * np.pi * self.n_vec)
        
        sigma = self.alpha + self.beta * omega**2
        ms = 0.25 * mu * self.Lx * Ly 
        
        P = (OutWeight * InWeight * self.k**2 * torch.exp(-sigma * self.k) / ms)
        
        # 1. Find only the valid indices (20Hz to 10kHz)
        valid_idx = torch.where((omega <= self.maxOm) & (omega >= (20 * 2 * np.pi)))[0]
        
        num_samples = int(self.sample_rate * duration)
        
        # Create a (1, T) row vector for the time indices
        n_row = torch.arange(num_samples, device=P.device, dtype=self.dtype).unsqueeze(0)
        
        # Initialize accumulator
        displacement_out = torch.zeros(num_samples, device=P.device, dtype=self.dtype)

        # 2. Process in chunks WITH Gradient Checkpointing
        chunk_size = 1000
        
        for i in range(0, len(valid_idx), chunk_size):
            idx_chunk = valid_idx[i:i + chunk_size]
            
            # Extract parameters for this chunk
            w_c = omega[idx_chunk].unsqueeze(1)
            s_c = sigma[idx_chunk].unsqueeze(1)
            p_c = P[idx_chunk].unsqueeze(1)
            
            # Call the method using self.compute_chunk
            chunk_out = checkpoint(self.compute_chunk, w_c, s_c, p_c, n_row, use_reentrant=False)
            
            # Accumulate (using + instead of += is safer for Autograd graph)
            displacement_out = displacement_out + chunk_out    
    
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