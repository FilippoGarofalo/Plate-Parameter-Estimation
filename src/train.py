import torch
import time

from model import DifferentiableModalPlate
from loss import TimeDomainEnergyLoss
from utils import load_target_audio, load_challenge_npz
from optimizer import get_optimizer

def main():
    # 1. SETUP & HYPERPARAMETERS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Replace this with the actual path to a challenge target IR
    #target_audio_path = "target/plate-ir.wav" 
    target_npz_path = "target/ground_truth_test.npz"
    sample_rate = 44100
    num_iterations = 2000
    LR = 0.01
    dtype = torch.float64   # switch to torch.float32 to halve memory and speed up at slight precision cost

    # Load the target audio
    #target_ir = load_target_audio(target_audio_path, target_sr=sample_rate, device=device, dtype=dtype, normalize=True)
    target_ir = load_challenge_npz(target_npz_path, device=device, dtype=dtype)

    # Computes target IR duration
    duration = len(target_ir) / sample_rate
    print(f"Target IR loaded: {len(target_ir)} samples ({duration:.2f} seconds)")

    sample_rate = 44100

    # 2. INITIALIZE MODULES
    model = DifferentiableModalPlate(sample_rate=sample_rate, plate_params=None, dtype=dtype).to(device)
    criterion = TimeDomainEnergyLoss(energy_weight=1.0, stft_weight=0.5).to(device)

    # Initialize Adam Optimizer
    # We use custom learning rates
    optimizer = get_optimizer(model, lr=LR)

    # 3. OPTIMIZATION LOOP
    print("\nStarting Optimization")
    start_time = time.time()
    
    for iteration in range(num_iterations):
        # Step 1: Clear the gradients
        optimizer.zero_grad()

        # Step 2: Forward Pass
        if iteration == 0: print("  [diag] forward...", flush=True)
        pred_ir = model(duration=duration, normalize=False, velCalc=False)

        # Step 3: Compute Loss
        if iteration == 0: print("  [diag] loss...", flush=True)
        loss = criterion(pred_ir, target_ir)

        # Step 4: Backward Pass
        if iteration == 0: print(f"  [diag] loss={loss.item():.6f}  backward...", flush=True)
        loss.backward()

        if iteration == 0:
            grad_norms = {n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None}
            print(f"  [diag] grad norms: {grad_norms}", flush=True)

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