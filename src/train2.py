import torch
import copy
import time
import numpy as np
from model import DifferentiableModalPlate
from loss import Loss
from loss2 import MSELoss
from utils import load_challenge_npz
from optimizer import get_optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lhs import lhs_sample_raw_params

def main():
    # ── HYPERPARAMETERS ───────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #target_npz_path = "target/2026-DATASET-STRIPPED/random_IR_0001.npz"
    target_npz_path = "target/ground_truth_test_1.npz"
    sample_rate     = 44100
    dtype           = torch.float64

    # LHS probe settings
    n_starts        = 10
    probe_iters     = 50
    lhs_seed        = 42
    LR_probe        = 0.01
    PROBE_DURATION  = 0.2

    # Phase 2 full optimization settings
    num_iterations  = 2500
    LR_stft         = 0.01
    LR_mse          = 0.01
    STFT_DURATION   = 1.0
    MSE_DURATION    = 0.5
    stft_switch_thr = 0.60   # switch to MSE when STFT loss drops below this

    # ── LOAD TARGET ───────────────────────────────────────────────────────────
    target_ir = load_challenge_npz(target_npz_path, device=device, dtype=dtype)
    duration  = len(target_ir) / sample_rate
    print(f"Target IR loaded: {len(target_ir)} samples ({duration:.2f} seconds)")

    criterion  = Loss(
        mse_weight=0.0, stft_weight=1.0, energy_weight=0.0,
        fft_sizes=[64, 128, 256, 1024]
    ).to(device)
    criterion2 = MSELoss().to(device)

    # ── PHASE 1: LHS PROBE ───────────────────────────────────────────────────
    lhs_params = lhs_sample_raw_params(n_starts, seed=lhs_seed)
    print(f"\nPhase 1 — probing {n_starts} LHS starts × {probe_iters} iters "
          f"(window={PROBE_DURATION}s)")

    best_probe_loss = float('inf')
    best_state_dict = None
    probe_target    = target_ir[:int(PROBE_DURATION * sample_rate)]
    phase1_start    = time.time()

    for start_idx, raw_params in enumerate(lhs_params):
        model     = DifferentiableModalPlate(sample_rate=sample_rate,
                                             plate_params=raw_params, dtype=dtype).to(device)
        optimizer = get_optimizer(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=LR_probe)
        criterion.precompute_target_stft(probe_target)

        for iteration in range(probe_iters):
            optimizer.zero_grad()
            pred_ir = model(duration=PROBE_DURATION, normalize=False, velCalc=False)
            loss    = criterion(pred_ir, probe_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            if iteration % 10 == 0:
                mu, D_mu, T0_mu, Ly, xo, yo = [
                    p.detach().cpu().item() for p in model.get_physical_parameters()
                ]
                print(f"    iter {iteration:03d} | loss: {loss.item():.6f} | "
                      f"mu: {mu:.4f} | D/mu: {D_mu:.6f} | T0/mu: {T0_mu:.6f} | "
                      f"Ly: {Ly:.4f} | xo: {xo:.4f} | yo: {yo:.4f}")

        probe_loss = loss.item()
        mu, D_mu, T0_mu, Ly, xo, yo = [
            p.detach().cpu().item() for p in model.get_physical_parameters()
        ]
        print(f"  Start {start_idx+1:02d}/{n_starts} | probe loss: {probe_loss:.6f} | "
              f"Ly: {Ly:.4f}m | xo: {xo:.4f}m | yo: {yo:.4f}m | "
              f"mu: {mu:.4f} | D/mu: {D_mu:.6f} | T0/mu: {T0_mu:.6f}")

        if probe_loss < best_probe_loss:
            best_probe_loss = probe_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            print(f"  >> New best (loss={best_probe_loss:.6f})")

        del model, optimizer, pred_ir, loss
        torch.cuda.empty_cache()

    print(f"\nPhase 1 done in {time.time()-phase1_start:.2f}s. "
          f"Best probe loss: {best_probe_loss:.6f}")

    # ── PHASE 2: FULL STFT → MSE OPTIMIZATION ────────────────────────────────
    print(f"\nPhase 2 — {num_iterations} iters from best start (STFT → MSE)")

    model = DifferentiableModalPlate(sample_rate=sample_rate, dtype=dtype).to(device)
    model.load_state_dict(best_state_dict)

    optimizer   = get_optimizer(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=LR_stft)
    scheduler   = ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                    patience=100, min_lr=1e-4)
    previous_lr = LR_stft

    use_mse        = False
    mse_start_iter = None

    progress = {
        'iteration': [], 'loss': [], 'phase': [],
        'mu': [], 'D_over_mu': [], 'T0_over_mu': [], 'Ly': [], 'xo': [], 'yo': []
    }

    start_time = time.time()
    for iteration in range(num_iterations):
        optimizer.zero_grad()

        if iteration == 0:
            print(" [diag] forward...", flush=True)

        # Curriculum
        if not use_mse:
            curr_duration = min(0.05 + (iteration / 500) * STFT_DURATION, STFT_DURATION)
        else:
            mse_elapsed   = iteration - mse_start_iter
            curr_duration = min(0.05 + (mse_elapsed / 500) * MSE_DURATION, MSE_DURATION)

        pred_ir           = model(duration=curr_duration, normalize=False, velCalc=False)
        target_ir_cropped = target_ir[:pred_ir.shape[0]]

        if not use_mse:
            criterion.precompute_target_stft(target_ir_cropped)
            loss = criterion(pred_ir, target_ir_cropped)
        else:
            loss = criterion2(pred_ir, target_ir_cropped)

        if iteration == 0:
            print(f" [diag] loss={loss.item():.6f} backward...", flush=True)

        loss.backward()

        if iteration == 0:
            grad_norms = {n: p.grad.norm().item()
                          for n, p in model.named_parameters() if p.grad is not None}
            print(f" [diag] grad norms: {grad_norms}", flush=True)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # STFT → MSE switch
        if not use_mse and loss.item() < stft_switch_thr:
            use_mse        = True
            mse_start_iter = iteration
            optimizer.param_groups[0]['lr'] = LR_mse
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                          patience=30, min_lr=1e-5)
            previous_lr = LR_mse
            print(f"\n {'='*58}")
            print(f"  [switch] Iter {iteration:04d} → MSE phase | "
                  f"loss={loss.item():.4f} | LR reset to {LR_mse}")
            print(f" {'='*58}\n")

        optimizer.zero_grad()

        # Scheduler
        scheduler.step(loss)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != previous_lr:
            phase_tag = 'mse' if use_mse else 'stft'
            print(f" [diag] LR → {current_lr:.2e}  (phase={phase_tag})")
            previous_lr = current_lr

        # Logging
        if iteration % 10 == 0 or iteration == num_iterations - 1:
            mu, D_mu, T0_mu, Ly, xo, yo = [
                p.detach().cpu().item() for p in model.get_physical_parameters()
            ]
            phase_tag = 'MSE' if use_mse else 'STFT'
            print(f"Iter {iteration:04d} [{phase_tag}] | Loss: {loss.item():.6f}")
            print(f"  Ly: {Ly:.4f}m | xo: {xo:.4f}m | yo: {yo:.4f}m | "
                  f"mu: {mu:.4f} | D/mu: {D_mu:.6f} | T0/mu: {T0_mu:.6f}")
            print("-" * 60)

            progress['iteration'].append(iteration)
            progress['loss'].append(loss.item())
            progress['phase'].append(phase_tag)
            progress['mu'].append(mu)
            progress['D_over_mu'].append(D_mu)
            progress['T0_over_mu'].append(T0_mu)
            progress['Ly'].append(Ly)
            progress['xo'].append(xo)
            progress['yo'].append(yo)

    total_time = time.time() - start_time
    print(f"\nOptimization complete in {total_time:.2f} seconds.")

    np.savez('target/train_progress.npz', **{k: np.array(v) for k, v in progress.items()})
    print("Training progress saved to target/train_progress.npz")

    mu, D_mu, T0_mu, Ly, xo, yo = [
        p.detach().cpu().item() for p in model.get_physical_parameters()
    ]
    print("\n=== FINAL ESTIMATED PARAMETERS ===")
    print(f"mu      := {mu:.6f}")
    print(f"D/mu    := {D_mu:.6f}")
    print(f"T0/mu   := {T0_mu:.6f}")
    print(f"Ly      := {Ly:.4f} m")
    print(f"xo      := {xo:.4f} m")
    print(f"yo      := {yo:.4f} m")
    print("==================================")

if __name__ == "__main__":
    main()
