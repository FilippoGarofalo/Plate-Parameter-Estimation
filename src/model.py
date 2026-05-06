import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

@torch.jit.script
def solve_modal_system(G1: torch.Tensor, G2: torch.Tensor, P: torch.Tensor, 
                       num_samples: int) -> torch.Tensor:
    
        num_modes = G1.shape[0]
        
        q1 = torch.zeros(num_modes, dtype=G1.dtype, device=G1.device)
        q2 = torch.zeros(num_modes, dtype=G1.dtype, device=G1.device)
        y = torch.zeros(num_samples, dtype=G1.dtype, device=G1.device)
        
        for n in range(num_samples):
            fin = 1.0 if n == 0 else 0.0
            
            q = G1 * q1 - G2 * q2 + P * fin
            
            y[n] = torch.sum(q1)
            
            q2 = q1.clone()
            q1 = q
        return y

class DifferentiableModalPlate(nn.Module):

    def __init__(self, sample_rate: int = 44100, plate_params: dict = None,
             dtype: torch.dtype = torch.float32):
        super(DifferentiableModalPlate, self).__init__()

        import math

        self.sample_rate = sample_rate
        self.k = 1.0 / sample_rate
        self.fmax = 10000.0
        self.maxOm = self.fmax * 2 * math.pi
        self.dtype = dtype

        # =========================
        # 1. FIXED PARAMETERS 
        # =========================
        self.register_buffer('Lx', torch.tensor(1.0, dtype=dtype))
        self.register_buffer('tau_0', torch.tensor(6.0, dtype=dtype))
        self.register_buffer('tau_1', torch.tensor(2.0, dtype=dtype))
        self.register_buffer('loss_f1', torch.tensor(500.0, dtype=dtype))

        # =========================
        # 2. RAYLEIGH DAMPING 
        # =========================
        OmDamp1 = torch.tensor(0.0, dtype=dtype)
        OmDamp2 = 2 * torch.pi * self.loss_f1

        dOmSq = OmDamp2**2 - OmDamp1**2

        eps = torch.tensor(1e-12, dtype=dtype)
        dOmSq = torch.clamp(dOmSq, min=eps)

        log10 = math.log(10.0)

        alpha = 3 * log10 / dOmSq * (OmDamp2**2 / self.tau_0 - OmDamp1**2 / self.tau_1)
        beta  = 3 * log10 / dOmSq * (1.0 / self.tau_1 - 1.0 / self.tau_0)

        self.register_buffer('alpha', alpha.clone().detach())
        self.register_buffer('beta', beta.clone().detach())

        # =========================
        # 3. LEARNABLE PARAMETERS
        # =========================
        def init_param(name, default):
            if plate_params is None:
                return nn.Parameter(torch.tensor(default, dtype=dtype))
            else:
                if name not in plate_params:
                    raise KeyError(f"Missing parameter '{name}' in plate_params")
                return nn.Parameter(torch.tensor(plate_params[name], dtype=dtype))

        self.mu_raw         = init_param('mu_raw', 0.5)
        self.D_over_mu_raw  = init_param('D_over_mu_raw', 0.5)
        self.T0_over_mu_raw = init_param('T0_over_mu_raw', 0.5)
        self.Ly_raw         = init_param('Ly_raw', 0.5)
        self.xo_raw         = init_param('xo_raw', 0.5)
        self.yo_raw         = init_param('yo_raw', 0.5)

    def get_physical_parameters(self):
        device = self.Lx.device
        dtype = self.dtype

        def to_norm(x):
            return torch.sigmoid(x)

        def map_range_linear(x, min_v, max_v):
            min_v = torch.as_tensor(min_v, dtype=dtype, device=device)
            max_v = torch.as_tensor(max_v, dtype=dtype, device=device)

            norm_x = to_norm(x)
            return min_v + norm_x * (max_v - min_v)

        def map_range_log(x, min_v, max_v):
            min_v = torch.as_tensor(min_v, dtype=dtype, device=device)
            max_v = torch.as_tensor(max_v, dtype=dtype, device=device)

            norm_x = to_norm(x)

            log_min = torch.log(min_v)
            log_max = torch.log(max_v)

            val_log = log_min + norm_x * (log_max - log_min)

            return torch.exp(val_log)

        # =========================
        # PARAMETRI FISICI
        # =========================
        mu = map_range_log(self.mu_raw, 2.43, 106.15)
        D_over_mu = map_range_log(self.D_over_mu_raw, 0.2805, 201.188)
        T0_over_mu = map_range_log(self.T0_over_mu_raw, 9.4e-5, 411.52)

        Ly = map_range_linear(self.Ly_raw, 1.1, 4.0)

        xo = map_range_linear(self.xo_raw, 0.51 * self.Lx, 1.0 * self.Lx)
        yo = map_range_linear(self.yo_raw, 0.51 * Ly, 1.0 * Ly)

        
        eps = torch.tensor(1e-12, dtype=dtype, device=device)

        mu = torch.clamp(mu, min=eps)
        D_over_mu = torch.clamp(D_over_mu, min=eps)
        T0_over_mu = torch.clamp(T0_over_mu, min=eps)
        Ly = torch.clamp(Ly, min=eps)

        return mu, D_over_mu, T0_over_mu, Ly, xo, yo
    
    def forward(self, duration: float = 1.0, normalize: bool = True, velCalc: bool = False) -> torch.Tensor:
        mu, D_over_mu, T0_over_mu, Ly, xo, yo = self.get_physical_parameters()

        device = self.Lx.device
        pi = torch.pi

        # =========================
        # 1. MODAL GRID 
        # =========================
        DDx = 120
        DDy = 120

        m_idx = torch.arange(1, DDx, device=device, dtype=self.dtype)
        n_idx = torch.arange(1, DDy, device=device, dtype=self.dtype)

        m_grid, n_grid = torch.meshgrid(m_idx, n_idx, indexing='ij')
        m_vec = m_grid.flatten()
        n_vec = n_grid.flatten()

        # =========================
        # 2. MODAL FREQUENCIES
        # =========================
        g1 = (m_vec * pi / self.Lx)**2 + (n_vec * pi / Ly)**2
        g2 = g1 * g1

        omega_sq = T0_over_mu * g1 + D_over_mu * g2
        omega = torch.sqrt(torch.clamp(omega_sq, min=0.0))

        # =========================
        # 3. APPLY LOW-FREQ RULE 
        # =========================
        #omega = torch.where(omega < 20 * 2 * pi, self.maxOm + 1000.0, omega)  #Removed as in the ModalPlate

        # =========================
        # 4. SORT 
        # =========================
        omega, sort_idx = torch.sort(omega, stable=True)
        m_vec = m_vec[sort_idx]
        n_vec = n_vec[sort_idx]

        # =========================
        # 5. TRUNCATE 
        # =========================
        valid = omega <= self.maxOm

        omega = omega[valid]
        m_vec = m_vec[valid]
        n_vec = n_vec[valid]

        # =========================
        # 6. DAMPING + COEFFICIENTI
        # =========================
        sigma = self.alpha + self.beta * omega**2

        exp_term = torch.exp(-sigma * self.k)

        G1 = 2 * torch.cos(omega * self.k) * exp_term
        G2 = exp_term * exp_term

        frac_xi = 0.335
        frac_yi = 0.467
        frac_xo = xo / self.Lx
        frac_yo = yo / Ly

        InWeight = torch.sin(frac_xi * pi * m_vec) * torch.sin(frac_yi * pi * n_vec)
        OutWeight = torch.sin(frac_xo * pi * m_vec) * torch.sin(frac_yo * pi * n_vec)

        ms = 0.25 * mu * self.Lx * Ly

        P = 4.0 * OutWeight * InWeight * self.k**2 * exp_term / (ms * self.Lx * Ly)

        # =========================
        # 7. TIME INTEGRATION 
        # =========================
        num_samples = int(self.sample_rate * duration)
        
        y = solve_modal_system(G1, G2, P, num_samples)


        if velCalc:
            y_prev_tensor = torch.cat((torch.tensor([0.0], device=device, dtype=self.dtype), y[:-1]))
            y = (y - y_prev_tensor) / self.k

        # =========================
        # 8. NORMALIZATION
        # =========================
        if normalize:
            peak = torch.max(torch.abs(y)) + 1e-8
            y = y / peak

        return y

        
