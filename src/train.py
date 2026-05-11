import torch
import time
import numpy as np
from model import DifferentiableModalPlate
from loss import Loss
from loss2 import MSELoss
from utils import load_challenge_npz
from optimizer import get_optimizer

def main():
    # 1. SETUP & HYPERPARAMETERS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    target_npz_path = "target/ground_truth_test_1.npz"
    sample_rate = 44100
    num_iterations = 1000
    LR = 0.01
    dtype = torch.float64

    target_ir = load_challenge_npz(target_npz_path, device=device, dtype=dtype)

    duration = len(target_ir) / sample_rate
    print(f"Target IR loaded: {len(target_ir)} samples ({duration:.2f} seconds)")

    # 2. INITIALIZE MODULES
    target_mu = 19.625000
    target_D_mu = 14.154282
    target_T0_mu = 22.929936
    Ly_target = 2.2500
    Lx_target = 1.0  # From your fixed params
    target_xo = 0.7500
    target_yo = 1.8450

    from utils import (inverse_map_softplus_linear, inverse_map_softplus_log, 
                       inverse_map_range_linear, inverse_map_range_log)

    perfect_initial_guess = {
        'mu_raw':         inverse_map_range_log(target_mu, 2.43, 106.15),
        'D_over_mu_raw':  inverse_map_range_log(target_D_mu, 0.2805, 201.188),
        'T0_over_mu_raw': inverse_map_range_log(target_T0_mu, 9.4e-5, 411.52),
        'Ly_raw':         inverse_map_range_linear(Ly_target, 1.1, 4.0),
        'xo_raw':         inverse_map_range_linear(target_xo, 0.51 * Lx_target, 1.0 * Lx_target),
        'yo_raw':         inverse_map_range_linear(target_yo, 0.51 * Ly_target, 1.0 * Ly_target),
    }

    # Initialize model with the perfect guess
    model = DifferentiableModalPlate(
        sample_rate=sample_rate, 
        plate_params=perfect_initial_guess, 
        dtype=dtype
    ).to(device)
    
    criterion = Loss(
        mse_weight=0.0,
        stft_weight=1.0,
        energy_weight=0.0,
        fft_sizes=[64, 128, 256, 1024]
       ).to(device)

    #model.Ly_raw.requires_grad = False
    #model.xo_raw.requires_grad = False
    #model.yo_raw.requires_grad = False

    active_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = get_optimizer(active_params ,lr=LR)
    
    # OPTIMIZATION: Precompute target STFT once (cached for all iterations)
    criterion.precompute_target_stft(target_ir)
    #criterion = MSELoss().to(device)
    progress = {'iteration': [], 'loss': [], 'mu': [], 'D_over_mu': [], 'T0_over_mu': [], 'Ly': [], 'xo': [], 'yo': []}

    # 3. OPTIMIZATION LOOP
    print("\nStarting Optimization")
    start_time = time.time()
    for iteration in range(num_iterations):
        # Step 1: Clear the gradients
        optimizer.zero_grad()

        # Step 2: Forward Pass
        if iteration == 0: 
            print(" [diag] forward...", flush=True)
        pred_ir = model(duration=duration, normalize=False, velCalc=False)
        if iteration == 0: 
            print(" [diag] loss...", flush=True)
        loss = criterion(pred_ir, target_ir)

        # Step 4: Backward Pass
        if iteration == 0: 
            print(f" [diag] loss={loss.item():.6f} backward...", flush=True)
        loss.backward()

        if iteration == 0:
            grad_norms = {n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None}
            print(f" [diag] grad norms: {grad_norms}", flush=True)

        # Step 6: Update Parameters
        optimizer.step()
        optimizer.zero_grad()

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

            progress['iteration'].append(iteration)
            progress['loss'].append(loss.item())
            progress['mu'].append(mu)
            progress['D_over_mu'].append(D_over_mu)
            progress['T0_over_mu'].append(T0_over_mu)
            progress['Ly'].append(Ly)
            progress['xo'].append(xo)
            progress['yo'].append(yo)

    total_time = time.time() - start_time
    print(f"\nOptimization complete in {total_time:.2f} seconds.")

    np.savez('target/train_progress.npz', **{k: np.array(v) for k, v in progress.items()})
    print("Training progress saved to target/train_progress.npz")

    # 4. RESULTS
    # ---------------------------------------------------------
    mu, D_over_mu, T0_over_mu, Ly, xo, yo = [
    p.detach().cpu().item() for p in model.get_physical_parameters()
    ]
    print("\n=== FINAL ESTIMATED PARAMETERS ===")
    print(f"mu := {mu:.6f}")
    print(f"D/mu := {D_over_mu:.6f}")
    print(f"T0/mu := {T0_over_mu:.6f}")
    print(f"Ly := {Ly:.4f} m")
    print(f"xo := {xo:.4f} m")
    print(f"yo := {yo:.4f} m")
    print("==================================")

if __name__ == "__main__":
    main()