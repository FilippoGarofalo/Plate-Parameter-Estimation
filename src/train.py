import torch
import time
import numpy as np
from model import DifferentiableModalPlate
from loss import Loss
from utils import load_challenge_npz
from optimizer import get_optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

def main():
    # 1. SETUP & HYPERPARAMETERS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    target_npz_path = "target/2026-DATASET-STRIPPED/random_IR_0001.npz"
    sample_rate     = 44100
    num_iterations  = 1500
    LR              = 0.01
    dtype           = torch.float64

    # Phase-switch settings
    plateau_patience = 50     # iters without improvement before switching to MSE
    plateau_delta    = 0.005  # improvement must exceed this to reset the counter
    LR_mse           = 0.005  # LR for MSE fine-tuning phase
    min_lr_mse       = 1e-5   # stop reducing below this in MSE phase

    target_ir = load_challenge_npz(target_npz_path, device=device, dtype=dtype)
    duration  = 1
    print(f"Target IR loaded: {len(target_ir)} samples ({duration:.2f} seconds)")

    model = DifferentiableModalPlate(sample_rate=sample_rate, dtype=dtype).to(device)

    criterion = Loss(
        mse_weight=0.0,
        stft_weight=1.0,
        energy_weight=0.0,
        fft_sizes=[64, 128, 256, 1024]
    ).to(device)

    optimizer   = get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler   = ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=50)
    previous_lr = LR

    progress = {
        'iteration': [], 'loss': [], 'phase': [],
        'mu': [], 'D_over_mu': [], 'T0_over_mu': [], 'Ly': [], 'xo': [], 'yo': []
    }

    phase            = 'stft'
    best_stft_loss   = float('inf')
    no_improve_count = 0

    print("\nStarting Optimization")
    start_time = time.time()

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        if iteration == 0:
            print(" [diag] forward...", flush=True)

        curr_duration     = min(0.05 + (iteration / 500) * duration, duration)
        pred_ir           = model(duration=curr_duration, normalize=False, velCalc=False)
        target_ir_cropped = target_ir[:pred_ir.shape[0]]

        if phase == 'stft':
            criterion.precompute_target_stft(target_ir_cropped)
        loss = criterion(pred_ir, target_ir_cropped)

        if iteration == 0:
            print(f" [diag] loss={loss.item():.6f} backward...", flush=True)

        loss.backward()

        if iteration == 0:
            grad_norms = {n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None}
            print(f" [diag] grad norms: {grad_norms}", flush=True)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        # ── Phase-switch: STFT → MSE ──────────────────────────────────────────
        if phase == 'stft':
            if loss.item() < best_stft_loss - plateau_delta:
                best_stft_loss   = loss.item()
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= plateau_patience:
                phase = 'mse'
                criterion.stft_weight = 0.0
                criterion.mse_weight  = 1.0
                optimizer   = get_optimizer(
                    filter(lambda p: p.requires_grad, model.parameters()), lr=LR_mse
                )
                scheduler   = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                patience=30, min_lr=min_lr_mse)
                previous_lr = LR_mse
                no_improve_count = 0
                print(f"\n {'='*58}")
                print(f"  [switch] Iter {iteration:04d} — STFT plateau "
                      f"(best={best_stft_loss:.4f}, current={loss.item():.4f})")
                print(f"  [switch] → MSE phase  |  LR reset to {LR_mse}")
                print(f" {'='*58}\n")

        # ── Scheduler step ────────────────────────────────────────────────────
        scheduler.step(loss)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != previous_lr:
            print(f" [diag] LR reduced to {current_lr:.2e}  (phase={phase})")
            previous_lr = current_lr

        # ── Logging ──────────────────────────────────────────────────────────
        if iteration % 10 == 0 or iteration == num_iterations - 1:
            mu, D_over_mu, T0_over_mu, Ly, xo, yo = [
                p.detach().cpu().item() for p in model.get_physical_parameters()
            ]
            print(f"Iter {iteration:04d} [{phase.upper():4s}] | Loss: {loss.item():.6f}")
            print(f"  Ly: {Ly:.4f}m | xo: {xo:.4f}m | yo: {yo:.4f}m | "
                  f"mu: {mu:.4f} | D/mu: {D_over_mu:.6f} | T0/mu: {T0_over_mu:.6f}")
            print("-" * 60)

            progress['iteration'].append(iteration)
            progress['loss'].append(loss.item())
            progress['phase'].append(phase)
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

    mu, D_over_mu, T0_over_mu, Ly, xo, yo = [
        p.detach().cpu().item() for p in model.get_physical_parameters()
    ]
    print("\n=== FINAL ESTIMATED PARAMETERS ===")
    print(f"mu      := {mu:.6f}")
    print(f"D/mu    := {D_over_mu:.6f}")
    print(f"T0/mu   := {T0_over_mu:.6f}")
    print(f"Ly      := {Ly:.4f} m")
    print(f"xo      := {xo:.4f} m")
    print(f"yo      := {yo:.4f} m")
    print("==================================")

if __name__ == "__main__":
    main()
