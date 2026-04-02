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

def test_forward_pass_equivalence():
    target_ir, test_ir = get_ir();

    error_signal = target_ir - test_ir;
    time_axis = np.linspace(0,duration,len(target_ir))
    print(f"Average Absolute Error: {np.mean(np.abs(error_signal)):.6f}")
    print(f"Max Absolute Error: {np.max(np.abs(error_signal)):.6f}")
    
    #plots
    plt.figure(figsize=(14, 10))

    # Plot 1: Full 1-second Impulse Response
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, target_ir, label='NumPy IR', alpha=0.7, color='blue')
    plt.plot(time_axis, test_ir, label='PyTorch IR', alpha=0.7, linestyle='--', color='orange')
    plt.title('Full Impulse Responses (1 Second)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 2: Zoomed-in view (first 50ms) to inspect transients
    plt.subplot(3, 1, 2)
    plt.plot(time_axis, target_ir, label='NumPy IR', alpha=0.7, color='blue', marker='o', markersize=2)
    plt.plot(time_axis, test_ir, label='PyTorch IR', alpha=0.7, linestyle='--', color='orange', marker='x', markersize=2)
    plt.xlim(0, 0.05) # Zooming in to the first 50 milliseconds
    plt.title('Zoomed In: First 50ms Transients')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 3: The isolated Error Signal
    plt.subplot(3, 1, 3)
    plt.plot(time_axis, error_signal, label='Error (NumPy - PyTorch)', color='red')
    plt.title('Error Difference Highlighted')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude Difference')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

    return

  

test_forward_pass_equivalence()