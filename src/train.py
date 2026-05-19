import torch
import time
import numpy as np
from model import DifferentiableModalPlate
from loss import Loss
from loss2 import MSELoss
from utils import load_challenge_npz
from optimizer import get_optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

def main():
    # 1. SETUP & HYPERPARAMETERS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    #target_npz_path = "target/ground_truth_test_1.2.npz"
    target_npz_path = "target/2026-DATASET-STRIPPED/random_IR_0001.npz"
    sample_rate = 44100
    num_iterations = 1500
    LR = 0.1
    dtype = torch.float64

    target_ir = load_challenge_npz(target_npz_path, device=device, dtype=dtype)

    duration = len(target_ir) / sample_rate
    print(f"Target IR loaded: {len(target_ir)} samples ({duration:.2f} seconds)")

    model = DifferentiableModalPlate(
        sample_rate=sample_rate,
        dtype=dtype
    ).to(device)
    
    criterion = Loss(
        mse_weight=0.0,
        stft_weight=1.0,
        energy_weight=0.0,
        fft_sizes=[64, 128, 256, 1024]
       ).to(device)
    
    criterion2 = MSELoss().to(device)

    active_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = get_optimizer(active_params ,lr=LR)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, min_lr=1e-3)
    previous_lr = LR
    

    #criterion = MSELoss().to(device)
    progress = {'iteration': [], 'loss': [], 'mu': [], 'D_over_mu': [], 'T0_over_mu': [], 'Ly': [], 'xo': [], 'yo': []}

    # 3. OPTIMIZATION LOOP
    print("\nStarting Optimization")
    start_time = time.time()
    curr_duration = 0.1
    for iteration in range(num_iterations):
        
        # Step 1: Clear the gradients
        optimizer.zero_grad()

        # Step 2: Forward Pass
        if iteration == 0: 
            print(" [diag] forward...", flush=True)

        #curr_duration = min(0.05 + (iteration / 700) * duration, duration-3.5)
        pred_ir = model(duration=curr_duration, normalize=False, velCalc=False)
        curr_samples = pred_ir.shape[0]
        target_ir_cropped = target_ir[:curr_samples]
        #
        if(criterion != criterion2):
            criterion.precompute_target_stft(target_ir_cropped)

        loss = criterion(pred_ir, target_ir_cropped)
        if iteration == 0: 
            print(" [diag] loss...", flush=True)
        #loss = criterion(pred_ir, target_ir)

        # Step 4: Backward Pass
        if iteration == 0: 
            print(f" [diag] loss={loss.item():.6f} backward...", flush=True)
        loss.backward()
        
        if iteration == 0:
            grad_norms = {n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None}
            print(f" [diag] grad norms: {grad_norms}", flush=True)

        # Step 6: Update Parameters
        optimizer.step()
        if criterion == criterion2 and loss.item() < 1:
            optimizer.param_groups[0]['lr'] = 0.01
            print(f" [diag] Switching to MSELoss and reducing LR to {0.001}", flush=True)

        if(loss.item() < 0.50):
            iter = iteration;
            criterion = criterion2;
            optimizer.param_groups[0]['lr'] = 0.1
            #curr_duration = min(0.1 + ((iteration-iter) / 500) * duration, duration-3.5)
            if(iteration % 10 == 0):
                print(f" [diag] Switching to MSELoss and reducing LR to {0.01}", flush=True)
        optimizer.zero_grad()

        # Step 6.5: Scheduler step
        # CRITICAL: ONLY step the scheduler after your progressive growing phase (iteration 200)
        # Otherwise, the growing signal duration will artificially trigger learning rate drops
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