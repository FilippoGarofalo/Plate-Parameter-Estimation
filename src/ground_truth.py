import numpy as np
import os
from ModalPlate import ModalPlate

# Physical bounds — must match model.py / lhs.py
MU_BOUNDS      = (2.43,    106.15)
D_BOUNDS       = (0.2805,  201.188)
T0_BOUNDS      = (9.4e-5,  411.52)
LY_BOUNDS      = (1.1,     4.0)
XO_FRAC_BOUNDS = (0.51,    1.0)
YO_FRAC_BOUNDS = (0.51,    1.0)

FIXED_RHO = 7850.0
FIXED_NU  = 0.25
FIXED_LX  = 1.0

PARAM_KEYS = ['mu', 'D_over_mu', 'T0_over_mu', 'Ly', 'xo', 'yo']


def _sample_physical_params(rng: np.random.Generator) -> dict:
    def log_sample(lo, hi):
        return np.exp(rng.uniform(np.log(lo), np.log(hi)))

    def lin_sample(lo, hi):
        return rng.uniform(lo, hi)

    mu         = log_sample(*MU_BOUNDS)
    D_over_mu  = log_sample(*D_BOUNDS)
    T0_over_mu = log_sample(*T0_BOUNDS)
    Ly         = lin_sample(*LY_BOUNDS)
    xo         = lin_sample(XO_FRAC_BOUNDS[0] * FIXED_LX, XO_FRAC_BOUNDS[1] * FIXED_LX)
    yo         = lin_sample(YO_FRAC_BOUNDS[0] * Ly,        YO_FRAC_BOUNDS[1] * Ly)

    return {'mu': mu, 'D_over_mu': D_over_mu, 'T0_over_mu': T0_over_mu,
            'Ly': Ly, 'xo': xo, 'yo': yo}


def _physical_to_plate_params(p: dict) -> dict:
    """Convert optimisation-space physical params to ModalPlate constructor dict."""
    h  = p['mu'] / FIXED_RHO
    # D_over_mu = E*h^2 / (12*(1-nu^2)*rho)  →  E = D_over_mu * 12*(1-nu^2)*rho / h^2
    E  = p['D_over_mu'] * 12 * (1 - FIXED_NU**2) * FIXED_RHO / (h**2)
    T0 = p['T0_over_mu'] * p['mu']

    return {
        'Lx':      FIXED_LX,
        'Ly':      p['Ly'],
        'h':       h,
        'T0':      T0,
        'rho':     FIXED_RHO,
        'E':       E,
        'nu':      FIXED_NU,
        'T60_DC':  6.0,
        'T60_F1':  2.0,
        'loss_F1': 500.0,
        'fp_x':    0.335,
        'fp_y':    0.467,
        'op_x':    p['xo'] / FIXED_LX,
        'op_y':    p['yo'] / p['Ly'],
    }


def generate_random_gt(filename: str = None, seed: int = None, duration: float = 5.0) -> tuple[str, dict]:
    """
    Sample random physical parameters, synthesise a GT IR and save to target/.
    The npz contains 'ir' plus 'gt_<param>' entries for all 6 physical params.

    Returns (save_path, gt_physical).
    """
    rng = np.random.default_rng(seed)
    gt = _sample_physical_params(rng)
    plate_params = _physical_to_plate_params(gt)

    if filename is None:
        tag = str(seed) if seed is not None else "rand"
        filename = f"ground_truth_random_{tag}.npz"

    print("--- GROUND TRUTH PARAMETERS ---")
    for k in PARAM_KEYS:
        print(f"  {k:12s}: {gt[k]:.6g}")
    print("--------------------------------")

    print(f"Synthesising IR ({duration} s) …")
    plate = ModalPlate(sample_rate=44100, plate_params=plate_params)
    plate.fmax = 10000.0
    plate.setup()
    ir = plate.synthesize_ir_method(duration=duration, normalize=False, velCalc=False)

    os.makedirs('target', exist_ok=True)
    save_path = os.path.join('target', filename)
    np.savez(save_path, ir=ir, **{f'gt_{k}': np.float64(gt[k]) for k in PARAM_KEYS})
    print(f"Saved: {save_path}")
    return save_path, gt


def compute_nmse(estimated: dict, gt_source) -> tuple[dict, float]:
    """
    Compute per-parameter and overall NMSE between estimated and GT params.

    gt_source: either a path to the .npz produced by generate_random_gt(),
               or a dict with the same keys as PARAM_KEYS.
    """
    if isinstance(gt_source, str):
        data = np.load(gt_source)
        gt = {k: float(data[f'gt_{k}']) for k in PARAM_KEYS}
    else:
        gt = gt_source

    nmse = {k: (estimated[k] - gt[k])**2 / (gt[k]**2 + 1e-12) for k in PARAM_KEYS}
    overall = float(np.mean(list(nmse.values())))

    print("\n=== NMSE Evaluation ===")
    print(f"{'Param':12s} {'GT':>14s} {'Estimated':>14s} {'NMSE':>12s}")
    print("-" * 56)
    for k in PARAM_KEYS:
        print(f"{k:12s} {gt[k]:14.6g} {estimated[k]:14.6g} {nmse[k]:12.4e}")
    print("-" * 56)
    print(f"{'Overall':12s} {overall:12.4e}")
    return nmse, overall


if __name__ == "__main__":
    generate_random_gt(seed=42, duration=5.0)
