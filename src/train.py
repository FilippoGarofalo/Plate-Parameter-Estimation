import torch
import torch.optim as optim
import time
import os

from model import DifferentiableModalPlate
from loss import MultiScaleSpectralLoss
from utils import load_target_audio
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

    # 2. INITIALIZE MODULES
    model = DifferentiableModalPlate(sample_rate=sample_rate).to(device)
    criterion = MultiScaleSpectralLoss().to(device)
    
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
        
        if iteration % 25 == 0 or iteration == num_iterations - 1:
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