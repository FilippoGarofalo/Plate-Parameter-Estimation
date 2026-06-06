import torch
import time
import numpy as np
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim import Adam
from model import DifferentiableModalPlate
from loss import Loss
from utils import load_challenge_npz
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ground_truth import compute_nmse
import pandas as pd
from pathlib import Path


# ── Minimal PSO (sequential, GPU-friendly, no logger dependency) ──────────────

def pso(cost_fn, bounds, num_particles=50, max_iter=200, w=0.7, c1=1.5, c2=1.5):
    """
    Minimise cost_fn over the box defined by bounds = [(lo, hi), ...].
    Returns (best_position, best_score, score_history).
    score_history[i] = global best score after iteration i.
    """
    dim   = len(bounds)
    lo    = np.array([b[0] for b in bounds], dtype=np.float64)
    hi    = np.array([b[1] for b in bounds], dtype=np.float64)
    span  = hi - lo

    # Initialise particles uniformly inside bounds
    particles  = lo + np.random.default_rng().random((num_particles, dim)) * span
    velocities = np.zeros_like(particles)

    personal_best        = particles.copy()
    personal_best_scores = np.array([cost_fn(p) for p in particles])

    global_best_idx   = int(np.argmin(personal_best_scores))
    global_best       = personal_best[global_best_idx].copy()
    global_best_score = personal_best_scores[global_best_idx]

    score_history = []

    for it in range(max_iter):
        r1 = np.random.default_rng().random((num_particles, dim))
        r2 = np.random.default_rng().random((num_particles, dim))

        velocities = (w * velocities
                      + c1 * r1 * (personal_best - particles)
                      + c2 * r2 * (global_best   - particles))
        particles += velocities
        particles  = np.clip(particles, lo, hi)

        scores = np.array([cost_fn(p) for p in particles])

        improved = scores < personal_best_scores
        personal_best[improved]        = particles[improved]
        personal_best_scores[improved] = scores[improved]

        best_idx = int(np.argmin(personal_best_scores))
        if personal_best_scores[best_idx] < global_best_score:
            global_best       = personal_best[best_idx].copy()
            global_best_score = personal_best_scores[best_idx]

        score_history.append(global_best_score)
        if it%10 == 0:
            print(f"  PSO iter {it+1:03d}/{max_iter} | best loss: {global_best_score:.4f}")

    return global_best, global_best_score, score_history


# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 1. SETUP & HYPERPARAMETERS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    target_npz_path = "target/2026-DATASET-STRIPPED/random_IR_0012.npz"
    sample_rate     = 44100
    num_iterations  = 1500
    dtype           = torch.float64

    # PSO Phase 1 settings
    PSO_PARTICLES = 50
    PSO_ITERS     = 100
    PSO_W, PSO_C1, PSO_C2 = 0.7, 1.5, 1.5
    PSO_DURATION  = 0.2   # seconds used in the PSO cost function

    # Raw-parameter bounds: sigmoid maps saturate past ±4, so ±4 covers full physical range
    RAW_BOUNDS = [(-4.0, 4.0)] * 6  # mu_raw, D_over_mu_raw, T0_over_mu_raw, Ly_raw, xo_raw, yo_raw
    RAW_KEYS   = ['mu_raw', 'D_over_mu_raw', 'T0_over_mu_raw', 'Ly_raw', 'xo_raw', 'yo_raw']

    target_stem  = Path(target_npz_path).stem
    target_index = target_stem.split('_')[-1] if '_' in target_stem else target_stem

    target_ir = load_challenge_npz(target_npz_path, device=device, dtype=dtype)
    duration  = len(target_ir) / sample_rate
    print(f"Target IR loaded: {len(target_ir)} samples ({duration:.2f} seconds)")

    criterion = Loss(
        mse_weight=0.0,
        stft_weight=1.0,
        energy_weight=0.0,
        fft_sizes=[64, 128, 256, 512, 1024, 2048, 4096],
    ).to(device)

    # ── PHASE 1: PSO exploration ───────────────────────────────────────────────
    print(f"\nPhase 1 — PSO exploration: {PSO_PARTICLES} particles, {PSO_ITERS} iterations")

    target_ir_pso = target_ir[:int(sample_rate * PSO_DURATION)]

    criterion_pso = Loss(
        mse_weight=0.0,
        stft_weight=1.0,
        energy_weight=0.0,
        fft_sizes=[256, 1024, 2048],
    ).to(device)
    criterion_pso.precompute_target_stft(target_ir_pso)

    def pso_cost(raw_vec):
        """Gradient-free cost: build model from raw params, forward pass, return STFT loss."""
        raw_params = dict(zip(RAW_KEYS, raw_vec.tolist()))
        with torch.no_grad():
            model_tmp = DifferentiableModalPlate(
                sample_rate=sample_rate,
                plate_params=raw_params,
                dtype=torch.float64,
            ).to(device)
            model_tmp.maxOm = 2500.0 * 2 * torch.pi
            pred = model_tmp(duration=PSO_DURATION, normalize=False, velCalc=False)
            loss_val = criterion_pso(pred, target_ir_pso).item()
        del model_tmp, pred
        torch.cuda.empty_cache()
        return loss_val

    best_raw_vec, best_pso_loss, pso_history = pso(
        pso_cost, RAW_BOUNDS,
        num_particles=PSO_PARTICLES,
        max_iter=PSO_ITERS,
        w=PSO_W, c1=PSO_C1, c2=PSO_C2,
    )

    best_raw_params = dict(zip(RAW_KEYS, best_raw_vec.tolist()))
    print(f"\n>>> Best PSO loss: {best_pso_loss:.4f}")
    print(">>> Best PSO parameters passed to Phase 2.")

    # Plot PSO convergence curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(range(1, len(pso_history) + 1), pso_history)
    ax.set_xlabel("PSO iteration")
    ax.set_ylabel("Best loss (log scale)")
    ax.set_title("Phase 1 — PSO convergence")
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    pso_plot_path = Path("experiment_results_taskA") / f"phase1_pso_curve_{target_index}.png"
    pso_plot_path.parent.mkdir(exist_ok=True)
    fig.savefig(pso_plot_path, dpi=150)
    plt.close(fig)
    print(f"PSO convergence curve saved to {pso_plot_path}")
    # ──────────────────────────────────────────────────────────────────────────

    # ── PHASE 2: Full gradient optimisation from best PSO start ───────────────
    print(f"\nPhase 2 — full optimisation for {num_iterations} iterations from best PSO start")

    model = DifferentiableModalPlate(
        sample_rate=sample_rate,
        plate_params=best_raw_params,
        dtype=dtype,
    ).to(device)

    optimizer = Adam([
        {'params': [model.mu_raw, model.Ly_raw, model.xo_raw, model.yo_raw], 'lr': 0.01},
        {'params': [model.D_over_mu_raw, model.T0_over_mu_raw],              'lr': 0.01},
    ])

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=500, min_lr=1e-4)

    progress = {'iteration': [], 'loss': [], 'mu': [], 'D_over_mu': [], 'T0_over_mu': [], 'Ly': [], 'xo': [], 'yo': []}

    STFT_DURATION      = 3.0
    best_loss_phase2   = float('inf')
    best_params_phase2 = None

    print("\nStarting Optimisation")
    start_time = time.time()

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        if iteration < 1000:
            curr_duration = min(0.05 + (iteration / 2000) * STFT_DURATION, STFT_DURATION)
        else:
            curr_duration = STFT_DURATION

        pred_ir = model(duration=curr_duration, normalize=False, velCalc=False)
        curr_samples      = pred_ir.shape[0]
        target_ir_cropped = target_ir[:curr_samples]
        criterion.precompute_target_stft(target_ir_cropped)

        loss = criterion(pred_ir, target_ir_cropped)

        scheduler.step(loss.item())

        if iteration % 10 == 0:
            print(f" [diag] iter {iteration}, loss={loss.item():.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        current_loss_val = loss.item()

        # Track best physical parameters only after full-duration phase starts (iteration >= 1000)
        if iteration >= 1000 and current_loss_val < best_loss_phase2:
            best_loss_phase2 = current_loss_val
            with torch.no_grad():
                _mu, _D, _T0, _Ly, _xo, _yo = model.get_physical_parameters()
                best_params_phase2 = {
                    'mu':         _mu.item(),
                    'D_over_mu':  _D.item(),
                    'T0_over_mu': _T0.item(),
                    'Ly':         _Ly.item(),
                    'xo':         _xo.item(),
                    'yo':         _yo.item(),
                }

        optimizer.zero_grad(set_to_none=True)
        del pred_ir, target_ir_cropped, loss
        gc.collect()
        torch.cuda.empty_cache()

        if iteration % 10 == 0 or iteration == num_iterations - 1:
            mu, D_over_mu, T0_over_mu, Ly, xo, yo = [
                p.detach().cpu().item() for p in model.get_physical_parameters()
            ]
            print(f"Iteration {iteration:04d} | Loss: {current_loss_val:.6f}")
            print(f"Ly: {Ly:.4f}m | xo: {xo:.4f}m | yo: {yo:.4f}m | "
                  f"mu: {mu:.4f} | D/mu: {D_over_mu:.6f} | T0/mu: {T0_over_mu:.6f}")
            print("-" * 60)

            progress['iteration'].append(iteration)
            progress['loss'].append(current_loss_val)
            progress['mu'].append(mu)
            progress['D_over_mu'].append(D_over_mu)
            progress['T0_over_mu'].append(T0_over_mu)
            progress['Ly'].append(Ly)
            progress['xo'].append(xo)
            progress['yo'].append(yo)

    total_time = time.time() - start_time
    print(f"\nOptimisation complete in {total_time:.2f} seconds.")

    np.savez('target/train_progress.npz', **{k: np.array(v) for k, v in progress.items()})

    # 4. RESULTS
    if best_params_phase2 is not None:
        print(f"\nUsing best Phase 2 parameters (loss={best_loss_phase2:.6f})")
        mu         = best_params_phase2['mu']
        D_over_mu  = best_params_phase2['D_over_mu']
        T0_over_mu = best_params_phase2['T0_over_mu']
        Ly         = best_params_phase2['Ly']
        xo         = best_params_phase2['xo']
        yo         = best_params_phase2['yo']
    else:
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

    npz_data  = np.load(target_npz_path)
    if 'gt_mu' in npz_data.files:
        estimated = {'mu': mu, 'D_over_mu': D_over_mu, 'T0_over_mu': T0_over_mu,
                     'Ly': Ly, 'xo': xo, 'yo': yo}
        compute_nmse(estimated, target_npz_path)
    else:
        print("(NMSE skipped: target file has no embedded ground truth params)")

    # ── Save results ───────────────────────────────────────────────────────────
    output_path = Path("experiment_results_taskA")
    output_path.mkdir(exist_ok=True)

    current_loss = round(best_loss_phase2 if best_params_phase2 is not None else progress['loss'][-1], 6)

    summary_row = {
        'target_file':       target_npz_path,
        'target_index':      target_index,
        'duration':          round(duration, 6),
        'optimization_time': round(total_time, 6),
        'best_loss':         current_loss,
        'iterations':        num_iterations,
    }
    summary_file = output_path / "experiment_summary.csv"
    new_row_df   = pd.DataFrame([summary_row])

    should_update = True
    if summary_file.exists():
        existing = pd.read_csv(summary_file)
        mask = existing['target_file'] == target_npz_path
        if mask.any():
            prev_loss = existing.loc[mask, 'best_loss'].values[0]
            if current_loss < prev_loss:
                existing = existing[~mask]
                pd.concat([existing, new_row_df], ignore_index=True).to_csv(summary_file, index=False)
                print(f"\nBetter run (loss {current_loss:.6f} < {prev_loss:.6f}), results updated")
            else:
                should_update = False
                print(f"\nPrevious run was better (loss {prev_loss:.6f} <= {current_loss:.6f}), keeping old results")
        else:
            pd.concat([existing, new_row_df], ignore_index=True).to_csv(summary_file, index=False)
            print(f"\nResults saved to {summary_file}")
    else:
        new_row_df.to_csv(summary_file, index=False)
        print(f"\nResults saved to {summary_file}")

    if should_update:
        best_params = {
            'mu':    mu,
            'D_mu':  D_over_mu,
            'T0_mu': T0_over_mu,
            'Ly':    Ly,
            'op_x':  xo,
            'op_y':  yo / Ly,
        }
        pd.DataFrame([best_params]).to_csv(
            output_path / f"best_params_{target_index}.csv",
            index=False, float_format='%.17g'
        )
        print(f"Parameters saved to best_params_{target_index}.csv")


if __name__ == "__main__":
    main()
