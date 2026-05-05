import torch
import time
from model import DifferentiableModalPlate
from loss import TimeDomainEnergyLoss
from utils import load_challenge_npz, invert_composite_parameters
from optimizer import get_optimizer

def main():
    # 1. SETUP & HYPERPARAMETERS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Replace this with the actual path to a challenge target IR
    target_npz_path = "target/ground_truth_test.npz"
    sample_rate = 44100
    num_iterations = 2000
    
    # Increased LR to 0.1 (Recommended if you switched to sigmoid normalization in [0,1])
    # If you are still using the unbounded tanh, you may need to lower this back to 0.01
    LR = 0.1 
    dtype = torch.float32

    # Target parameters for reference
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
    
    # Load the target audio
    target_ir = load_challenge_npz(target_npz_path, device=device, dtype=dtype)

    # Computes target IR duration
    duration = len(target_ir) / sample_rate
    print(f"Target IR loaded: {len(target_ir)} samples ({duration:.2f} seconds)")

    # 2. INITIALIZE MODULES
    model = DifferentiableModalPlate(sample_rate=sample_rate, plate_params=None, dtype=dtype).to(device)
    
    # Configure the loss to use Multi-Scale Spectral (MSS) and Energy only.
    # We set lowpass_weight=0.0 because the large FFT windows in MSS already handle the low frequencies.
    criterion = TimeDomainEnergyLoss(
        mse_weight=0.5, 
        stft_weight=10.0,      # Scales the MSS loss
        lowpass_weight=0.0,    # Disabled
        energy_weight=1.0, 
        fft_sizes=[64, 256, 1024, 4096]
    ).to(device)

    # Optimizer
    optimizer = get_optimizer(model, lr=LR)

    # 3. OPTIMIZATION LOOP
    print("\nStarting Optimization")
    start_time = time.time()
    
    for iteration in range(num_iterations):
        if iteration == 300:
            print("\n  >>> PHASE 2: Unlocking Geometry <<<", flush=True)
            model.Ly_raw.requires_grad = True
            model.xo_raw.requires_grad = True
            model.yo_raw.requires_grad = True
            
            # Re-initialize Adam so it grabs the newly unlocked parameters
            # We slightly drop the LR for fine-tuning
            optimizer = torch.optim.Adam(model.parameters(), lr=LR * 0.5)
        # Step 1: Clear the gradients
        optimizer.zero_grad()

        # Step 2: Forward Pass
        if iteration == 0: print("  [diag] forward...", flush=True)
        pred_ir = model(duration=duration, normalize=False, velCalc=False)
        
        # Step 3: Compute Loss (Curriculum learning removed; MSS handles it all)
        if iteration == 0: print("  [diag] loss...", flush=True)
        loss = criterion(pred_ir, target_ir)

        # Step 4: Backward Pass
        if iteration == 0: print(f"  [diag] loss={loss.item():.6f}  backward...", flush=True)
        loss.backward()

        # Step 5: Gradient Clipping (Crucial for stability with Adam and physical parameters)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if iteration == 0:
            grad_norms = {n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None}
            print(f"  [diag] grad norms: {grad_norms}", flush=True)

        # Step 6: Update Parameters
        optimizer.step()

        # Step 7: Print logs and parameter progress
        if iteration % 10 == 0 or iteration == num_iterations - 1:
            
            # Safely extract the current bounded physical values
            mu, D_over_mu, T0_over_mu, Ly, xo, yo = [
                p.detach().cpu().item() for p in model.get_physical_parameters()
            ]
            
            print(f"Iteration {iteration:04d} | Loss: {loss.item():.6f}")
            print(f"Ly: {Ly:.4f}m | xo: {xo:.4f}m | yo: {yo:.4f}m | "
                  f"mu: {mu:.4f} | D/mu: {D_over_mu:.6f} | T0/mu: {T0_over_mu:.6f}")
            
            print("-" * 60)

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