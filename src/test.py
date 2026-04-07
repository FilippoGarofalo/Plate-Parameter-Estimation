import torch
import numpy as np
import matplotlib.pyplot as plt
from model import DifferentiableModalPlate
from ModalPlate import ModalPlate
from utils import inverse_sigmoid, inverse_softplus

import importlib.util
import sys

sample_rate = 44100
duration = 1.0
# ---------------------------------------------------------
# A. Define the raw physical properties from ModalPlate.py
# ---------------------------------------------------------
Lx = 0.5
Ly = 1.1
h = 0.001
T0 = 0.01
rho = 2430.0
E = 6.7e10
nu = 0.25

# ---------------------------------------------------------
# B. Calculate the exact physical targets
# ---------------------------------------------------------
target_mu = rho * h
target_D = (E * h**3) / (12 * (1 - nu**2))

target_D_mu = target_D / target_mu
target_T0_mu = T0 / target_mu

target_xo = 0.61 * Lx
target_yo = 0.61 * Ly

# ---------------------------------------------------------
# C. Programmatically generate the raw PyTorch parameters
# ---------------------------------------------------------
perfect_initial_guess = {
    # Unbounded parameters (Inverse Softplus)
    'mu_raw': inverse_softplus(target_mu - 1e-4),
    'D_over_mu_raw': inverse_softplus(target_D_mu - 1e-4),
    'T0_over_mu_raw': inverse_softplus(target_T0_mu - 1e-4),

    # Bounded parameters (Inverse Sigmoid with exact boundaries from your model.py)
    'Ly_raw': inverse_sigmoid(Ly, 1.1, 4.0),
    
    'xo_raw': inverse_sigmoid(target_xo, 
                              0.49 * Lx, 
                              1.0 * Lx),
    
    'yo_raw': inverse_sigmoid(target_yo, 0.51 * Ly, 1.0 * Ly)
}

def get_ir():
    np_plate = ModalPlate(sample_rate=sample_rate);
    target_ir = np_plate.synthesize_ir_method(duration=duration, velCalc=False, normalize=True);
    torch_plate = DifferentiableModalPlate(sample_rate=sample_rate, plate_params=perfect_initial_guess);
    test_ir = torch_plate.forward(normalize=True, velCalc=False);
    test_ir = test_ir.detach().numpy();
    return target_ir, test_ir

def cumulative_energy(ir: np.ndarray) -> np.ndarray:
    """E_cum[n] = sum_{k=0}^{n} x[k]^2  (normalizzata al valore finale)"""
    energy = np.cumsum(ir ** 2)
    return energy / (energy[-1] + 1e-12)


def mse_cumulative_energy(ir_a: np.ndarray, ir_b: np.ndarray) -> float:
    return float(np.mean((cumulative_energy(ir_a) - cumulative_energy(ir_b)) ** 2))


def test_forward_pass_equivalence():
    target_ir, test_ir = get_ir()

    error_signal = target_ir - test_ir
    time_axis    = np.linspace(0, duration, len(target_ir))

    cum_target = cumulative_energy(target_ir)
    cum_test   = cumulative_energy(test_ir)
    mse_cum    = mse_cumulative_energy(target_ir, test_ir)
    mse_raw    = float(np.mean(error_signal ** 2))
    max_abs_err = float(np.max(np.abs(error_signal)))   # <-- aggiunto

    print(f"Average Absolute Error : {np.mean(np.abs(error_signal)):.6f}")
    print(f"Max Absolute Error     : {max_abs_err:.6f}")
    print(f"MSE (raw IR)           : {mse_raw:.2e}")
    print(f"MSE (cumulative energy): {mse_cum:.2e}")

    fig, axes = plt.subplots(4, 1, figsize=(14, 14))

    # 1 — IR completa
    axes[0].plot(time_axis, target_ir, label='NumPy IR',   alpha=0.7, color='blue')
    axes[0].plot(time_axis, test_ir,   label='PyTorch IR', alpha=0.7, color='orange', linestyle='--')
    axes[0].set_title('Full Impulse Responses (1 s)')
    axes[0].set_xlabel('Time [s]'); axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3); axes[0].legend()

    # Box metriche nel primo plot
    metrics_text = (
        f"MSE (raw): {mse_raw:.2e}\n"
        f"Max Abs Error: {max_abs_err:.2e}\n"
    )
    axes[0].text(
        0.02, 0.98, metrics_text,
        transform=axes[0].transAxes,
        va='top', ha='left',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
    )

    # 2 — Zoom primi 50 ms
    axes[1].plot(time_axis, target_ir, label='NumPy IR',   alpha=0.7, color='blue',   marker='o', markersize=2)
    axes[1].plot(time_axis, test_ir,   label='PyTorch IR', alpha=0.7, color='orange', marker='x', markersize=2, linestyle='--')
    axes[1].set_xlim(0, 0.05)
    axes[1].set_title('Zoomed: First 50 ms Transients')
    axes[1].set_xlabel('Time [s]'); axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3); axes[1].legend()

    # 3 — Energia cumulativa
    axes[2].plot(time_axis, cum_target, label='NumPy  cumulative energy', color='blue')
    axes[2].plot(time_axis, cum_test,   label='PyTorch cumulative energy', color='orange', linestyle='--')
    axes[2].fill_between(time_axis, cum_target, cum_test, alpha=0.15, color='red',
                         label=f'Gap  (MSE={mse_cum:.2e})')
    axes[2].set_title('Normalized Cumulative Energy')
    axes[2].set_xlabel('Time [s]'); axes[2].set_ylabel('Cumulative energy (normalized)')
    axes[2].grid(True, alpha=0.3); axes[2].legend()

    # 4 — Errore segnale grezzo
    axes[3].plot(time_axis, error_signal, label='Error (NumPy − PyTorch)', color='red')
    axes[3].set_title(f'Error Signal')
    axes[3].set_xlabel('Time [s]'); axes[3].set_ylabel('Amplitude difference')
    axes[3].grid(True, alpha=0.3); axes[3].legend()

    plt.tight_layout()
    plt.show()
  

test_forward_pass_equivalence()