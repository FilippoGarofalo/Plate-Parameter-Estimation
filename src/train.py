import torch
import torch.optim as optim
import time

from model import DifferentiableModalPlate
from loss import TimeDomainEnergyLoss
from utils import load_target_audio, inverse_softplus, inverse_sigmoid
from optimizer import get_optimizer

def main():
    # 1. SETUP & HYPERPARAMETERS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Replace this with the actual path to a challenge target IR
    target_audio_path = "target/plate-ir.wav" 
    
    sample_rate = 44100
    num_iterations = 300
    
    # Load the target audio
    target_ir = load_target_audio(target_audio_path, target_sr=sample_rate, device=device)
    
    # Computes target IR duration
    duration = len(target_ir) / sample_rate
    print(f"Target IR loaded: {len(target_ir)} samples ({duration:.2f} seconds)")

    sample_rate = 44100
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
        # Raw parameters learn normalized O(1) values — divide by scale before inverse_softplus
        'mu_raw': inverse_softplus(target_mu / DifferentiableModalPlate._MU_SCALE - 1e-4),
        'D_over_mu_raw': inverse_softplus(target_D_mu / DifferentiableModalPlate._D_MU_SCALE - 1e-4),
        'T0_over_mu_raw': inverse_softplus(target_T0_mu / DifferentiableModalPlate._T0_MU_SCALE - 1e-4),

        # Bounded parameters (Inverse Sigmoid with exact boundaries from your model.py)
        'Ly_raw': inverse_sigmoid(Ly, 1.1, 4.0),
        
        'xo_raw': inverse_sigmoid(target_xo, 
                                0.49 * Lx, 
                                1.0 * Lx),
        
        'yo_raw': inverse_sigmoid(target_yo, 0.51 * Ly, 1.0 * Ly)
    }
    # 2. INITIALIZE MODULES
    model = DifferentiableModalPlate(sample_rate=sample_rate, plate_params=perfect_initial_guess).to(device)
    criterion = TimeDomainEnergyLoss().to(device)

    # Initialize Adam Optimizer
    # We use custom learning rates
    optimizer = get_optimizer(model)

    # 3. OPTIMIZATION LOOP
    print("\nStarting Optimization")
    start_time = time.time()
    
    for iteration in range(num_iterations):
        # Step 1: Clear the gradients
        optimizer.zero_grad()
        
        # Step 2: Forward Pass
        pred_ir = model(duration=duration, normalize=True, velCalc=False)
        
        # Step 3: Compute Loss
        loss = criterion(pred_ir, target_ir)
        
        # Step 4: Backward Pass
        loss.backward()
        
        # Step 5: Update Parameters
        optimizer.step()
        
        # Step 6: Print the updated parameters (look ups for starting values in model.py)
        if iteration % 25 == 0 or iteration == num_iterations - 1:
            
            # Safely extract the current bounded physical values
            mu, D_over_mu, T0_over_mu, Ly, xo, yo = [
                p.detach().cpu().item() for p in model.get_physical_parameters()
            ]
            
            print(f"Ly: {Ly:.4f}m | xo: {xo:.4f}m | yo: {yo:.4f}m | "
                  f"mu: {mu:.4f} | D/mu: {D_over_mu:.6f} | T0/mu: {T0_over_mu:.6f}")
            # Iteration and loss logs
            print(f"Iteration {iteration:04d} | Loss: {loss.item():.6f}")
            

    total_time = time.time() - start_time
    print(f"\nOptimization complete in {total_time:.2f} seconds.")

    # 4. RESULTS
    # ---------------------------------------------------------
    mu, D_over_mu, T0_over_mu, Ly, xo, yo = [
        p.detach().cpu().item() for p in model.get_physical_parameters()
    ]
    
    print("\n=== FINAL ESTIMATED PARAMETERS ===")
    print(f"mu         := {mu:.6f}")
    print(f"D/mu       := {D_over_mu:.6f}")
    print(f"T0/mu      := {T0_over_mu:.6f}")
    print(f"Ly         := {Ly:.4f} m")
    print(f"xo         := {xo:.4f} m")
    print(f"yo         := {yo:.4f} m")
    print("==================================")

if __name__ == "__main__":
    main()