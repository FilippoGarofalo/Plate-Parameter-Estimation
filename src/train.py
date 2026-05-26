import torch
import time
import copy  ### MODIFIED: Added missing import ###
import numpy as np
from model import DifferentiableModalPlate
from loss import Loss
from utils import load_challenge_npz
from optimizer import get_optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lhs import lhs_sample_raw_params_2d

def main():
    # 1. SETUP & HYPERPARAMETERS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #target_npz_path = "target/ground_truth_test_1.1.npz"
    target_npz_path = "target/2026-DATASET-STRIPPED/random_IR_0002.npz"
    sample_rate     = 44100
    num_iterations  = 1000
    LR              = 0.1
    dtype           = torch.float64

    # Multi-start settings
    n_starts        = 100  
    lhs_seed        = 42

    ### MODIFIED: Bumped to 0.2 so 4096 and 8192 FFT sizes don't crash
    PHASE1_DURATION = 0.5  
    ### END MODIFIED ###

    target_ir = load_challenge_npz(target_npz_path, device=device, dtype=dtype)

    duration = len(target_ir) / sample_rate
    print(f"Target IR loaded: {len(target_ir)} samples ({duration:.2f} seconds)")

    criterion = Loss(
        mse_weight=0.0,
        stft_weight=1.0,
        energy_weight=0.0,
        fft_sizes=[64, 128, 256, 1024, 4096]
    ).to(device)
    #criterion = MSELoss().to(device)  # Start with MSE for simplicity in Phase 1

    # ── PHASE 1: ZERO-SHOT PROBING ────────────────────────────
    lhs_params = lhs_sample_raw_params_2d(n_starts, seed=lhs_seed)
    print(f"\nPhase 1 — Zero-shot probing {n_starts} LHS starts (Ultra-fast, No Gradients)")

    best_probe_loss  = float('inf')
    best_state_dict  = None
    phase1_start     = time.time()

    # Precompute the probe target ONCE for all 100 samples
    probe_target = target_ir[:int(PHASE1_DURATION * sample_rate)]
    criterion.precompute_target_stft(probe_target)

    for start_idx, raw_params in enumerate(lhs_params):
        model = DifferentiableModalPlate(
            sample_rate=sample_rate,
            plate_params=raw_params,
            dtype=dtype
        ).to(device)

        # ⚡ CRITICAL FIX: No gradients! Just evaluate the physics once.
        with torch.no_grad():
            pred_ir = model(duration=PHASE1_DURATION, normalize=False, velCalc=False)
            probe_loss = criterion(pred_ir, probe_target).item()

        # Get physical parameters for logging
        mu, D_over_mu, T0_over_mu, Ly, xo, yo = [
            p.cpu().item() for p in model.get_physical_parameters()
        ]
        
        # Only print every 10th start to keep the console clean, OR if we find a new best
        if probe_loss < best_probe_loss:
            best_probe_loss = probe_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            
            print(f"  Start {start_idx + 1:02d}/{n_starts} | probe loss: {probe_loss:.6f} | "
                  f"Ly: {Ly:.4f}m | xo: {xo:.4f}m | yo: {yo:.4f}m | "
                  f"mu: {mu:.4f} | D/mu: {D_over_mu:.6f} | T0/mu: {T0_over_mu:.6f}")
            print(f"  >> New best probe (loss={best_probe_loss:.6f})")

        # Free GPU memory
        del model, pred_ir
        torch.cuda.empty_cache()

    phase1_time = time.time() - phase1_start
    print(f"\nPhase 1 done in {phase1_time:.2f}s. Best probe loss: {best_probe_loss:.6f}")

    # ── PHASE 2: Full optimization from best start ────────────────
    print(f"\nPhase 2 — full optimization for {num_iterations} iterations from best start")

    model = DifferentiableModalPlate(
        sample_rate=sample_rate,
        dtype=dtype
    ).to(device)
    model.load_state_dict(best_state_dict)


    ### MODIFIED: Cleaned up duplicate declarations ###
    criterion = Loss(
        mse_weight=0.0,
        stft_weight=1.0,
        energy_weight=0.0,
        fft_sizes=[64, 128, 256, 1024, 4096]
    ).to(device)
    active_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_optimizer(active_params, lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, min_lr=1e-4)
    previous_lr = LR
    
    progress = {'iteration': [], 'loss': [], 'mu': [], 'D_over_mu': [], 'T0_over_mu': [], 'Ly': [], 'xo': [], 'yo': []}

    STFT_DURATION = 2.0

    # 3. OPTIMIZATION LOOP
    print("\nStarting Optimization")
    start_time = time.time()
    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Step 2: Forward Pass
        if iteration == 0: 
            print(" [diag] forward...", flush=True)

        curr_duration = min(0.05 + (iteration/500)*STFT_DURATION, STFT_DURATION)

        pred_ir = model(duration=curr_duration, normalize=False, velCalc=False)
        curr_samples = pred_ir.shape[0]
        target_ir_cropped = target_ir[:curr_samples]

        criterion.precompute_target_stft(target_ir_cropped)
        loss = criterion(pred_ir, target_ir_cropped)

        if iteration == 0: 
            print(" [diag] loss...", flush=True)
            print(f" [diag] loss={loss.item():.6f} backward...", flush=True)

        if iteration % 10 == 0:
            _lv = loss.item()
            _ls = f"{_lv:.2e}" if _lv < 1e-3 else f"{_lv:.4f}"
            print(f" [diag] iter {iteration}, loss={_ls}, lr={optimizer.param_groups[0]['lr']:.6f}")
            
        # Step 4: Backward Pass
        loss.backward()
        
        if iteration == 0:
            grad_norms = {n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None}
            print(f" [diag] grad norms: {grad_norms}", flush=True)
        
        # Step 5b: Clip gradients to prevent sigmoid saturation blow-ups
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Step 6: Update Parameters
        optimizer.step()
        
        # ── Scheduler step ──
        scheduler.step(loss.item())
        if iteration % 10 == 0:
            print(f" [diag] STFT phase: iter {iteration}, loss={loss.item():.4f}, "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}")
        
        
        # Step 7: Print logs and parameter progress
        if iteration % 10 == 0 or iteration == num_iterations - 1:
            mu, D_over_mu, T0_over_mu, Ly, xo, yo = [
            p.detach().cpu().item() for p in model.get_physical_parameters()
            ]
            loss_val = loss.item()
            loss_str = f"{loss_val:.2e}" if loss_val < 1e-3 else f"{loss_val:.6f}"
            print(f"Iteration {iteration:04d} | Loss: {loss_str}")
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