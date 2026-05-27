import numpy as np
from scipy.stats import qmc
from utils import (
    inverse_map_sigm_linear,
    inverse_map_sigm_log,
    inverse_map_softplus_linear,
    inverse_map_softplus_log
)

# Physical bounds — must match model.py get_physical_parameters()
MU_BOUNDS      = (2.43,   106.15)
D_BOUNDS       = (0.2805, 201.188)
T0_BOUNDS      = (1, 411.52)
LY_BOUNDS      = (1.1,    4.0)
XO_FRAC_BOUNDS = (0.51,   1.0)   # xo as fraction of Lx=1.0
YO_FRAC_BOUNDS = (0.51,   1.0)   # yo as fraction of Ly

T0_WEIGHT = 0.1  # weight used in map_sigm_log for T0_over_mu_raw

def lhs_sample_raw_params_2d(n_starts, seed=42):
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    unit_samples = sampler.random(n=n_starts)

    def log_interp(u, lo, hi):
        return np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))

    raw_params_list = []
    for u in unit_samples:
        D_over_mu  = log_interp(u[0], *D_BOUNDS)
        T0_over_mu = log_interp(u[1], *T0_BOUNDS)
        mu         = log_interp(u[2], *MU_BOUNDS)

        D_over_mu_raw  = inverse_map_softplus_log(D_over_mu, *D_BOUNDS)
        T0_over_mu_raw = inverse_map_softplus_log(T0_over_mu, *T0_BOUNDS, scale=T0_WEIGHT)
        mu_raw         = inverse_map_softplus_log(mu, *MU_BOUNDS)

        raw_params_list.append({
            'mu_raw':         float(mu_raw),    # default → middle of range
            'D_over_mu_raw':  float(D_over_mu_raw),
            'T0_over_mu_raw': float(T0_over_mu_raw),
            'Ly_raw':         0.0,    # default
            'xo_raw':         0.0,    # default
            'yo_raw':         0.0,    # default
        })
    return raw_params_list

def lhs_sample_raw_params(n_starts: int, seed: int = 42) -> list[dict]:
    """
    Generate n_starts raw-parameter starting points via Latin Hypercube Sampling.

    Samples are drawn uniformly in [0,1]^6, mapped to physical space
    (log-scale for mu / D_over_mu / T0_over_mu, linear for Ly / xo / yo),
    then inverted to the raw unconstrained values that DifferentiableModalPlate expects.

    Returns a list of dicts with keys:
        mu_raw, D_over_mu_raw, T0_over_mu_raw, Ly_raw, xo_raw, yo_raw
    """
    sampler = qmc.LatinHypercube(d=6, seed=seed)
    unit_samples = sampler.random(n=n_starts)  # [n_starts, 6]

    def log_interp(u, lo, hi):
        return np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))

    def lin_interp(u, lo, hi):
        return lo + u * (hi - lo)

    raw_params_list = []
    for u in unit_samples:
        # Map to physical space
        mu         = log_interp(u[0], *MU_BOUNDS)
        D_over_mu  = log_interp(u[1], *D_BOUNDS)
        T0_over_mu = log_interp(u[2], *T0_BOUNDS)
        Ly         = lin_interp(u[3], *LY_BOUNDS)
        xo         = lin_interp(u[4], *XO_FRAC_BOUNDS)  # absolute, Lx=1.0
        yo         = lin_interp(u[5], *YO_FRAC_BOUNDS) * Ly

        # Invert to raw space
        mu_raw         = inverse_map_softplus_log(mu,         *MU_BOUNDS)
        D_over_mu_raw  = inverse_map_softplus_log(D_over_mu,  *D_BOUNDS)
        T0_over_mu_raw = inverse_map_softplus_log(T0_over_mu, *T0_BOUNDS, scale=T0_WEIGHT)
        Ly_raw         = inverse_map_sigm_linear(Ly, *LY_BOUNDS)
        xo_raw         = inverse_map_sigm_linear(xo, *XO_FRAC_BOUNDS)
        yo_raw         = inverse_map_sigm_linear(yo, 0.51 * Ly, Ly)

        raw_params_list.append({
            'mu_raw':         float(mu_raw),
            'D_over_mu_raw':  float(D_over_mu_raw),
            'T0_over_mu_raw': float(T0_over_mu_raw),
            'Ly_raw':         float(Ly_raw),
            'xo_raw':         float(xo_raw),
            'yo_raw':         float(yo_raw),
        })

    return raw_params_list
