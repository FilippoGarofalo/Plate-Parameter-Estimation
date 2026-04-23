import numpy as np
import os
from ModalPlate import ModalPlate

def generate_custom_target(filename="ground_truth_test.npz"):
    
    gt_params = {
        'Lx': 1.0,           # Fisso
        'Ly': 2.25,          # [1.1, 4.0]
        'h': 0.0025,         # [0.001, 0.005]
        'T0': 450.0,         # [0.01, 1000.0]
        'rho': 7850.0,       # [2430.0, 21230.0] (Acciaio)
        'E': 20.0e10,        # [6.7e10, 22.0e10]
        'nu': 0.25,          # Fisso
        'T60_DC': 6.0,       # Fisso
        'T60_F1': 2.0,       # Fisso
        'loss_F1': 500.0,    # Fisso
        'fp_x': 0.335,       # Fisso
        'fp_y': 0.467,       # Fisso
        'op_x': 0.75,        # [0.51, 1.0]
        'op_y': 0.82         # [0.51, 1.0]
    }


    mu = gt_params['rho'] * gt_params['h']
    D = (gt_params['E'] * gt_params['h']**3) / (12 * (1 - gt_params['nu']**2))
    
    target_mu = mu
    target_D_mu = D / mu
    target_T0_mu = gt_params['T0'] / mu
    target_xo = gt_params['op_x'] * gt_params['Lx']
    target_yo = gt_params['op_y'] * gt_params['Ly']

    print("--- GROUND TRUTH TARGETS FOR EVALUATION ---")
    print(f"mu      : {target_mu:.6f}")
    print(f"D/mu    : {target_D_mu:.6f}")
    print(f"T0/mu   : {target_T0_mu:.6f}")
    print(f"Ly      : {gt_params['Ly']:.4f}")
    print(f"xo      : {target_xo:.4f}")
    print(f"yo      : {target_yo:.4f}")
    print("-------------------------------------------\n")

    print("Synthesizing IR (5 seconds)...")
    plate = ModalPlate(sample_rate=44100, plate_params=gt_params)
    ir = plate.synthesize_ir_method(duration=5.0, normalize=False, velCalc=False)

    os.makedirs('target', exist_ok=True)
    save_path = os.path.join('target', filename)
    
    np.savez(save_path, ir=ir)
    print(f"File salvato con successo: {save_path}")

if __name__ == "__main__":
    generate_custom_target()