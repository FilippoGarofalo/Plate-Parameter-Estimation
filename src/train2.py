import torch
import copy
import time
import numpy as np
from model import DifferentiableModalPlate
from loss import Loss
from utils import load_challenge_npz
from optimizer import get_optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lhs import lhs_sample_raw_params

def main():
    # 1. SETUP & HYPERPARAMETERS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    target_npz_path = "target/ground_truth_test_1.npz"
    sample_rate     = 44100
    num_iterations  = 1000
    LR              = 0.01
    dtype           = torch.float64

    # Multi-start settings
    n_starts        = 20    
    probe_iters     = 50   # short run per LHS start to find best basin
    lhs_seed        = 42

    PHASE1_DURATION = 0.5  # fixed short window for LHS probing phase
    PHASE2_DURATION = 1 # full target duration for final optimization phase

    target_ir = load_challenge_npz(target_npz_path, device=device, dtype=dtype)

    duration = len(target_ir) / sample_rate
    print(f"Target IR loaded: {len(target_ir)} samples ({duration:.2f} seconds)")

    criterion = Loss(
        mse_weight=0.0,
        stft_weight=1.0,
        energy_weight=0.0,
        fft_sizes=[64, 128, 256, 1024, 4096]
    ).to(device)

    # ── PHASE 1: Probe all LHS starts ────────────────────────────
    lhs_params = lhs_sample_raw_params(n_starts, seed=lhs_seed)
    print(f"\nPhase 1 — probing {n_starts} LHS starts for {probe_iters} iterations each")

    best_probe_loss  = float('inf')
    best_state_dict  = None
    phase1_start     = time.time()

    for start_idx, raw_params in enumerate(lhs_params):
        model = DifferentiableModalPlate(
            sample_rate=sample_rate,
            plate_params=raw_params,
            dtype=dtype
        ).to(device)

        active_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = get_optimizer(active_params, lr=LR)

        for iteration in range(probe_iters):
            optimizer.zero_grad()

            curr_duration = min(0.05 + (iteration / 200) * PHASE1_DURATION, PHASE1_DURATION) # fixed short window for probing
            pred_ir = model(duration=curr_duration, normalize=False, velCalc=False)
            curr_samples = pred_ir.shape[0]
            target_ir_cropped = target_ir[:curr_samples]

            criterion.precompute_target_stft(target_ir_cropped)
            loss = criterion(pred_ir, target_ir_cropped)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            mu, D_over_mu, T0_over_mu, Ly, xo, yo = [
                p.detach().cpu().item() for p in model.get_physical_parameters()
            ]
            if iteration % 10 == 0:
                print(f"    iter {iteration:03d} | loss: {loss.item():.6f} | "
                    f"mu: {mu:.4f} | D/mu: {D_over_mu:.6f} | T0/mu: {T0_over_mu:.6f} | "
                    f"Ly: {Ly:.4f} | xo: {xo:.4f} | yo: {yo:.4f}")

        probe_loss = loss.item()
        mu, D_over_mu, T0_over_mu, Ly, xo, yo = [
            p.detach().cpu().item() for p in model.get_physical_parameters()
        ]
        print(f"  Start {start_idx + 1:02d}/{n_starts} | probe loss: {probe_loss:.6f} | "
              f"Ly: {Ly:.4f}m | xo: {xo:.4f}m | yo: {yo:.4f}m | "
              f"mu: {mu:.4f} | D/mu: {D_over_mu:.6f} | T0/mu: {T0_over_mu:.6f}")

        if probe_loss < best_probe_loss:
            best_probe_loss = probe_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            print(f"  >> New best probe (loss={best_probe_loss:.6f})")

        # Explicitly free GPU memory before the next start
        del model, optimizer, pred_ir, loss, target_ir_cropped
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

    active_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer  = get_optimizer(active_params, lr=0.01)
    scheduler  = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    previous_lr = LR

    criterion.precompute_target_stft(target_ir)
    progress = {'iteration': [], 'loss': [], 'mu': [], 'D_over_mu': [], 'T0_over_mu': [], 'Ly': [], 'xo': [], 'yo': []}

    start_time = time.time()
    for iteration in range(num_iterations):
        # Step 1: Clear the gradients
        optimizer.zero_grad()

        # Step 2: Forward Pass
        if iteration == 0:
            print(" [diag] forward...", flush=True)

        curr_duration = min(0.05 + (iteration / 200) * PHASE2_DURATION, PHASE2_DURATION)
        pred_ir = model(duration=curr_duration, normalize=False, velCalc=False)
        curr_samples = pred_ir.shape[0]
        target_ir_cropped = target_ir[:curr_samples]

        criterion.precompute_target_stft(target_ir_cropped)
        loss = criterion(pred_ir, target_ir_cropped)
        if iteration == 0:
            print(" [diag] loss...", flush=True)

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

        # Step 6.5: Scheduler step
        # CRITICAL: ONLY step the scheduler after your progressive growing phase (iteration 200)
        # Otherwise, the growing signal duration will artificially trigger learning rate drops
        if iteration >= 200 and loss.item() <= 0.30:
            scheduler.step(loss)
            # Print when the learning rate changes so you can monitor the drops
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr != previous_lr:
                print(f" [diag] Plateau hit! Learning Rate reduced to: {current_lr}")
                previous_lr = current_lr

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
