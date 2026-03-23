import pytest
import torch
import numpy as np

import importlib.util
import sys


def import_libs():
    # ----------------- ----------------------------------------
    # 1. Import original ModalPlate from explicit file path
    # ---------------------------------------------------------
    modal_path = "adjust_your_path/ModalPlate.py"
    spec_modal = importlib.util.spec_from_file_location("ModalPlate_module", modal_path)
    modal_module = importlib.util.module_from_spec(spec_modal)
    sys.modules["ModalPlate_module"] = modal_module
    spec_modal.loader.exec_module(modal_module)

    # Extract the class from the loaded file
    ModalPlate = modal_module.ModalPlate

    # ---------------------------------------------------------
    # 2. Import PyTorch DifferentiableModalPlate from explicit file path
    # ---------------------------------------------------------
    model_path = "adjust_your_path/model.py"
    spec_model = importlib.util.spec_from_file_location("model_module", model_path)
    model_module = importlib.util.module_from_spec(spec_model)
    sys.modules["model_module"] = model_module
    spec_model.loader.exec_module(model_module)

    # Extract the class from the loaded file
    DifferentiableModalPlate = model_module.DifferentiableModalPlate

    # Extract the class from the loaded file
    DifferentiableModalPlate = model_module.DifferentiableModalPlate

    print("Successfully imported both classes via importlib!")
    
    # ADD THIS LINE: Return the loaded classes so other functions can use them
    return ModalPlate, DifferentiableModalPlate

    print("Successfully imported both classes via importlib!")

def inverse_softplus(y: float) -> float:
    """Calculates the raw variable needed to output 'y' after softplus."""
    # We subtract the 1e-4 epsilon we added in the model
    y_adj = y - 1e-4
    if y_adj <= 0:
        raise ValueError("Target value too small for softplus mapping.")
    return np.log(np.exp(y_adj) - 1.0)

def inverse_sigmoid(y: float, min_val: float, max_val: float) -> float:
    """Calculates the raw variable needed to output 'y' after scaled sigmoid."""
    norm_y = (y - min_val) / (max_val - min_val)
    if norm_y <= 0.0 or norm_y >= 1.0:
        raise ValueError("Target value strictly outside sigmoid bounds.")
    return np.log(norm_y / (1.0 - norm_y))

def test_forward_pass_equivalence():
    ModalPlate, DifferentiableModalPlate = import_libs()  # Ensure classes are imported before running the test
    # ---------------------------------------------------------
    # 1. DEFINE SHARED CHALLENGE PARAMETERS
    # ---------------------------------------------------------
    # Challenge fixed parameters
    Lx = 1.0
    nu = 0.25
    
    # Target physical properties we want to test
    Ly = 1.5
    h = 0.001
    rho = 2430.0
    E = 6.7e10
    T0 = 0.01
    
    # Input/Output positions (NumPy expects relative [0-1], PyTorch computes absolute [m])
    fp_x_rel = 0.335
    fp_y_rel = 0.467
    op_x_rel = 0.61
    op_y_rel = 0.75
    
    xo = op_x_rel * Lx
    yo = op_y_rel * Ly

    # ---------------------------------------------------------
    # 2. RUN ORIGINAL NUMPY MODEL
    # ---------------------------------------------------------
    np_params = {
        'Lx': Lx, 'Ly': Ly, 'h': h, 'rho': rho, 'E': E, 'nu': nu, 'T0': T0,
        'T60_DC': 6.0, 'T60_F1': 2.0, 'loss_F1': 500.0,
        'fp_x': fp_x_rel, 'fp_y': fp_y_rel,
        'op_x': op_x_rel, 'op_y': op_y_rel
    }
    
    np_plate = ModalPlate(sample_rate=44100, plate_params=np_params)
    
    # Generate 0.1 seconds of un-normalized audio
    duration = 0.1
    audio_np = np_plate.synthesize_ir_method(duration=duration, normalize=False)

    # ---------------------------------------------------------
    # 3. SET UP PYTORCH MODEL TO MATCH EXACTLY
    # ---------------------------------------------------------
    # Calculate lumped parameters required by PyTorch math
    mu = rho * h
    D = E * h**3 / (12.0 * (1.0 - nu**2))
    D_over_mu = D / mu
    T0_over_mu = T0 / mu

    pt_plate = DifferentiableModalPlate(sample_rate=44100)
    
    # Force the Lx buffer to match our test constraint
    pt_plate.Lx.data = torch.tensor(Lx, dtype=torch.float32)
    
    # Use inverse functions to set the exact _raw variables 
    # so the model's get_physical_parameters() yields our targets.
    with torch.no_grad():
        pt_plate.mu_raw.data = torch.tensor(inverse_softplus(mu), dtype=torch.float32)
        pt_plate.D_over_mu_raw.data = torch.tensor(inverse_softplus(D_over_mu), dtype=torch.float32)
        pt_plate.T0_over_mu_raw.data = torch.tensor(inverse_softplus(T0_over_mu), dtype=torch.float32)
        
        pt_plate.Ly_raw.data = torch.tensor(inverse_sigmoid(Ly, 1.1, 4.0), dtype=torch.float32)
        pt_plate.xo_raw.data = torch.tensor(inverse_sigmoid(xo, 0.49 * Lx, Lx), dtype=torch.float32)
        pt_plate.yo_raw.data = torch.tensor(inverse_sigmoid(yo, 0.51 * Ly, Ly), dtype=torch.float32)

    # Generate 0.5 seconds of un-normalized audio
    audio_pt = pt_plate(duration=duration, normalize=False)
    
    # Convert PyTorch output back to numpy array
    audio_pt_np = audio_pt.detach().numpy()

    # ---------------------------------------------------------
    # 4. ASSERTION AND METRICS
    # ---------------------------------------------------------
    assert audio_np.shape == audio_pt_np.shape, "Length of audio outputs do not match!"
    
    # Calculate the maximum absolute difference between the waveforms
    max_diff = np.max(np.abs(audio_np - audio_pt_np))
    print(f"\nMax difference between outputs: {max_diff:.8e}")
    
    # Check if the outputs are identical within a float32 tolerance threshold
    # Note: NumPy uses float64 by default, PyTorch uses float32. 
    # An absolute tolerance (atol) of 1e-4 is standard for cross-precision DSP testing.
    assert np.allclose(audio_np, audio_pt_np, atol=1e-4), \
        "Waveforms diverge! The mathematical translation has an error."
    


  

test_forward_pass_equivalence()