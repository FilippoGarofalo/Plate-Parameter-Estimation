import numpy as np
import matplotlib.pyplot as plt
from model import DifferentiableModalPlate
from ModalPlate import ModalPlate
from utils import inverse_map_range_linear, inverse_map_range_log, inverse_map_softplus_linear, inverse_map_softplus_log

sample_rate = 44100
duration = 1.0

Lx = 1.0          
Ly = 2.25         
h = 0.0025
T0 = 450.0
rho = 7850.0
E = 20.0e10
nu = 0.25

target_mu = rho * h
target_D = (E * h**3) / (12 * (1 - nu**2))
target_D_mu = target_D / target_mu
target_T0_mu = T0 / target_mu

target_xo = 0.75 * Lx  
target_yo = 0.82 * Ly  


custom_params = {
    'Lx': Lx,
    'Ly': Ly,
    'h': h,
    'T0': T0,
    'rho': rho,
    'E': E,
    'nu': nu,
    'T60_DC': 6.0,
    'T60_F1': 2.0,       
    'loss_F1': 500.0,
    'fp_x': 0.335,       
    'fp_y': 0.467,       
    'op_x': target_xo / Lx,
    'op_y': target_yo / Ly
}


perfect_initial_guess = {
    'mu_raw':         inverse_map_softplus_log(target_mu, 2.43, 106.15),
    'D_over_mu_raw':  inverse_map_softplus_log(target_D_mu, 0.2805, 201.188),
    'T0_over_mu_raw': inverse_map_softplus_log(target_T0_mu, 0.000094, 411.52),

    'Ly_raw': inverse_map_range_linear(Ly,        1.1,       4.0),
    'xo_raw': inverse_map_range_linear(target_xo, 0.51 * Lx, 1.0 * Lx),
    'yo_raw': inverse_map_range_linear(target_yo, 0.51 * Ly, 1.0 * Ly),
}

def get_ir():
    np_plate = ModalPlate(sample_rate=sample_rate, plate_params=custom_params)
    
    np_plate.fmax = 10000.0  
    np_plate.setup() 
    
    target_ir = np_plate.synthesize_ir_method(duration=duration, velCalc=False, normalize=True)
    
    torch_plate = DifferentiableModalPlate(sample_rate=sample_rate, plate_params=perfect_initial_guess)
    test_ir = torch_plate.forward(normalize=True, velCalc=False)
    test_ir = test_ir.detach().numpy()
    
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
    max_abs_err = float(np.max(np.abs(error_signal)))   

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