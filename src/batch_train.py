"""
batch_train.py
--------------
Runs the full two-phase optimisation (LHS probe → STFT → MSE) on every
random_IR_0001 … random_IR_0016 in the dataset, then saves:

  results/
    summary.csv              ← one row per IR: final loss + 6 estimated params
    training_results.png     ← single figure with loss curves + final params
"""

import os
import sys
import copy
import csv
import time
import glob

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(__file__))
from model     import DifferentiableModalPlate
from loss      import Loss
from loss2     import MSELoss
from utils     import load_challenge_npz
from optimizer import get_optimizer
from lhs       import lhs_sample_raw_params   # full 6-D LHS


# ══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
SAMPLE_RATE     = 44100
DTYPE           = torch.float64

N_STARTS        = 100
LHS_SEED        = 42
PHASE1_DURATION = 0.2

NUM_ITERATIONS  = 1000
LR              = 0.01
MAX_GRAD_NORM   = 1.0

STFT_DURATION   = 1.0
STFT_PATIENCE   = 150
STFT_MIN_DELTA  = 1e-4

DATASET_DIR     = "target/2026-DATASET-STRIPPED"
RESULTS_DIR     = "results"
# ══════════════════════════════════════════════════════════════════════════════


def train_one(target_npz_path: str, device: torch.device) -> dict:
    """
    Run Phase-1 (LHS zero-shot probe) + Phase-2 (STFT→MSE) on one IR.
    Returns a result dict including the per-iteration log for plotting.
    """
    tag = os.path.splitext(os.path.basename(target_npz_path))[0]
    print(f"\n{'='*70}\n  {tag}\n{'='*70}")

    # ── load ──────────────────────────────────────────────────────────────────
    target_ir    = load_challenge_npz(target_npz_path, device=device, dtype=DTYPE)
    duration     = len(target_ir) / SAMPLE_RATE
    MSE_DURATION = min(duration - 0.05, 1.0)   # cap at 1s — avoids OOM from huge mode×sample tensors
    print(f"  {len(target_ir)} samples  ({duration:.2f}s)  MSE cap={MSE_DURATION:.2f}s")

    criterion  = Loss(mse_weight=0.0, stft_weight=1.0, energy_weight=0.0,
                      fft_sizes=[64, 128, 256, 1024, 4096]).to(device)
    criterion2 = MSELoss().to(device)

    # ── Phase 1: zero-shot LHS probe ─────────────────────────────────────────
    lhs_params   = lhs_sample_raw_params(N_STARTS, seed=LHS_SEED)
    probe_target = target_ir[:int(PHASE1_DURATION * SAMPLE_RATE)]
    criterion.precompute_target_stft(probe_target)

    best_probe_loss = float('inf')
    best_state_dict = None
    t0 = time.time()

    for start_idx, raw_params in enumerate(lhs_params):
        model = DifferentiableModalPlate(sample_rate=SAMPLE_RATE,
                                         plate_params=raw_params, dtype=DTYPE).to(device)
        with torch.no_grad():
            pred_ir    = model(duration=PHASE1_DURATION, normalize=False, velCalc=False)
            probe_loss = criterion(pred_ir, probe_target).item()

        if probe_loss < best_probe_loss:
            best_probe_loss = probe_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            mu, D_mu, T0_mu, Ly, xo, yo = [
                p.cpu().item() for p in model.get_physical_parameters()]
            print(f"  [P1] {start_idx+1:03d}/{N_STARTS}  loss={probe_loss:.4f}  "
                  f"Ly={Ly:.3f} xo={xo:.3f} yo={yo:.3f} "
                  f"mu={mu:.3f} D/mu={D_mu:.5f} T0/mu={T0_mu:.5f}")
        del model, pred_ir
        torch.cuda.empty_cache()

    print(f"  Phase 1: {time.time()-t0:.1f}s  best={best_probe_loss:.4f}")

    # ── Phase 2: STFT → MSE ───────────────────────────────────────────────────
    model = DifferentiableModalPlate(sample_rate=SAMPLE_RATE, dtype=DTYPE).to(device)
    model.load_state_dict(best_state_dict)

    optimizer   = get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler   = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-4)
    previous_lr = LR

    use_mse        = False
    mse_start_iter = None
    mse_start_dur  = None   # duration at the moment of phase switch
    switch_iter    = None
    best_stft_loss  = float('inf')
    stft_no_improve = 0
    curr_dur        = 0.05  # tracked as state so MSE can continue from where STFT left off

    log = {k: [] for k in ['iter', 'loss', 'phase']}

    t1 = time.time()
    for iteration in range(NUM_ITERATIONS):
        optimizer.zero_grad()

        if not use_mse:
            curr_dur = min(0.05 + (iteration / NUM_ITERATIONS) * STFT_DURATION, STFT_DURATION)
        else:
            # grow from the window size at switch time → MSE_DURATION, never jump
            elapsed  = iteration - mse_start_iter
            curr_dur = min(mse_start_dur + (elapsed / 500) * (MSE_DURATION - mse_start_dur),
                           MSE_DURATION)

        pred_ir           = model(duration=curr_dur, normalize=False, velCalc=False)
        target_ir_cropped = target_ir[:pred_ir.shape[0]]

        if not use_mse:
            criterion.precompute_target_stft(target_ir_cropped)
            loss = criterion(pred_ir, target_ir_cropped)
        else:
            loss = criterion2(pred_ir, target_ir_cropped)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()

        # plateau-based phase switch
        if not use_mse:
            if loss.item() < best_stft_loss - STFT_MIN_DELTA:
                best_stft_loss  = loss.item()
                stft_no_improve = 0
            else:
                stft_no_improve += 1

            if stft_no_improve >= STFT_PATIENCE:
                use_mse        = True
                mse_start_iter = iteration
                mse_start_dur  = curr_dur   # continue from current window, no jump
                switch_iter    = iteration
                torch.cuda.empty_cache()
                for pg in optimizer.param_groups:
                    pg['lr'] = 0.01
                scheduler   = ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                patience=20, min_lr=1e-5)
                previous_lr = 0.01
                print(f"  [switch] iter {iteration:04d} → MSE  "
                      f"best_stft={best_stft_loss:.4f}  dur={curr_dur:.3f}s→{MSE_DURATION:.2f}s")

        optimizer.zero_grad()
        scheduler.step(loss.item())

        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != previous_lr:
            print(f"  [lr] {previous_lr:.2e} → {current_lr:.2e}  "
                  f"({'MSE' if use_mse else 'STFT'})")
            previous_lr = current_lr

        if iteration % 10 == 0 or iteration == NUM_ITERATIONS - 1:
            log['iter'].append(iteration)
            log['loss'].append(loss.item())
            log['phase'].append('MSE' if use_mse else 'STFT')

        if iteration % 200 == 0 or iteration == NUM_ITERATIONS - 1:
            mu, D_mu, T0_mu, Ly, xo, yo = [
                p.detach().cpu().item() for p in model.get_physical_parameters()]
            phase_tag = 'MSE' if use_mse else 'STFT'
            print(f"  [{phase_tag}] iter {iteration:04d}  loss={loss.item():.5f}  "
                  f"Ly={Ly:.3f} xo={xo:.3f} yo={yo:.3f} "
                  f"mu={mu:.3f} D/mu={D_mu:.5f} T0/mu={T0_mu:.5f}")

    print(f"  Phase 2: {time.time()-t1:.1f}s")
    mu, D_mu, T0_mu, Ly, xo, yo = [
        p.detach().cpu().item() for p in model.get_physical_parameters()]

    return {
        'tag':         tag,
        'final_loss':  float(np.array(log['loss'])[-1]),
        'switch_iter': switch_iter,
        'mu':    mu,  'D_mu':  D_mu,  'T0_mu': T0_mu,
        'Ly':    Ly,  'xo':    xo,    'yo':    yo,
        'log':   log,                  # kept in memory for the final plot
    }


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE OUTPUT FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def save_results(all_results: list[dict], results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    n = len(all_results)
    short_tags = [r['tag'].replace('random_IR_', 'IR_') for r in all_results]

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(results_dir, 'summary.csv')
    fields   = ['tag', 'final_loss', 'switch_iter', 'mu', 'D_mu', 'T0_mu', 'Ly', 'xo', 'yo']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_results)
    print(f"\nCSV → {csv_path}")

    # ── figure layout ─────────────────────────────────────────────────────────
    # Row A : loss curves  (4 cols × ceil(n/4) rows)
    # Row B : final-loss bar chart
    # Row C : 6 parameter bar charts (2 rows × 3 cols)

    ncols_curves = 4
    nrows_curves = int(np.ceil(n / ncols_curves))

    fig = plt.figure(figsize=(22, 5 * nrows_curves + 4 + 8))
    outer = gridspec.GridSpec(3, 1, figure=fig,
                              height_ratios=[nrows_curves * 5, 4, 8],
                              hspace=0.45)

    # ── A: loss curves grid ───────────────────────────────────────────────────
    gs_curves = gridspec.GridSpecFromSubplotSpec(
        nrows_curves, ncols_curves, subplot_spec=outer[0],
        hspace=0.55, wspace=0.35)

    for i, r in enumerate(all_results):
        ax  = fig.add_subplot(gs_curves[i // ncols_curves, i % ncols_curves])
        log = r['log']
        iters  = np.array(log['iter'])
        losses = np.array(log['loss'])
        phases = np.array(log['phase'])

        stft_m = phases == 'STFT'
        mse_m  = phases == 'MSE'

        if stft_m.any():
            ax.plot(iters[stft_m], losses[stft_m], color='steelblue',  lw=1.2)
        if mse_m.any():
            ax.plot(iters[mse_m],  losses[mse_m],  color='darkorange', lw=1.2)
        if r['switch_iter'] is not None:
            ax.axvline(r['switch_iter'], color='red', ls='--', lw=0.8)

        ax.set_title(short_tags[i], fontsize=8, fontweight='bold')
        ax.set_xlabel('iter', fontsize=7)
        ax.set_ylabel('loss', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
        ax.annotate(f"final={r['final_loss']:.3f}",
                    xy=(0.97, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=6.5,
                    color='black',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    # hide unused curve panels
    for i in range(n, nrows_curves * ncols_curves):
        fig.add_subplot(gs_curves[i // ncols_curves, i % ncols_curves]).set_visible(False)

    # ── B: final loss bar chart ───────────────────────────────────────────────
    ax_bar = fig.add_subplot(outer[1])
    final_losses = [r['final_loss'] for r in all_results]
    colors = ['steelblue' if l < np.median(final_losses) else 'darkorange'
              for l in final_losses]
    bars = ax_bar.bar(range(n), final_losses, color=colors, edgecolor='white', linewidth=0.5)
    ax_bar.axhline(np.mean(final_losses), color='red', ls='--', lw=1.2,
                   label=f'mean = {np.mean(final_losses):.3f}')
    ax_bar.set_xticks(range(n))
    ax_bar.set_xticklabels(short_tags, rotation=40, ha='right', fontsize=8)
    ax_bar.set_ylabel('Final loss')
    ax_bar.set_title('Final loss per IR  (blue ≤ median, orange > median)', fontsize=10)
    ax_bar.legend(fontsize=8)
    ax_bar.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, final_losses):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(final_losses) * 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=6.5)

    # ── C: parameter summary grid ─────────────────────────────────────────────
    param_keys   = ['mu',   'D_mu',   'T0_mu',  'Ly',  'xo',  'yo']
    param_labels = ['μ',    'D/μ',    'T₀/μ',   'Ly',  'xo',  'yo']

    gs_params = gridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=outer[2], hspace=0.55, wspace=0.35)

    for idx, (key, label) in enumerate(zip(param_keys, param_labels)):
        ax = fig.add_subplot(gs_params[idx // 3, idx % 3])
        vals = [r[key] for r in all_results]
        ax.bar(range(n), vals, color='steelblue', edgecolor='white', linewidth=0.5)
        ax.set_xticks(range(n))
        ax.set_xticklabels(short_tags, rotation=40, ha='right', fontsize=7)
        ax.set_title(label, fontsize=9, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(labelsize=7)

    fig.suptitle('Plate Parameter Estimation — Batch Results', fontsize=14, fontweight='bold', y=1.002)

    # add a legend for phase colours (top-right of figure)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='steelblue',  lw=2, label='STFT phase'),
        Line2D([0], [0], color='darkorange', lw=2, label='MSE phase'),
        Line2D([0], [0], color='red', lw=1.5, ls='--', label='phase switch'),
    ]
    fig.legend(handles=legend_elements, loc='upper right',
               fontsize=8, framealpha=0.8, ncol=3)

    out_path = os.path.join(results_dir, 'training_results.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    npz_files = sorted(glob.glob(os.path.join(DATASET_DIR, "random_IR_*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {DATASET_DIR}")
    print(f"Found {len(npz_files)} target IRs")

    all_results = []
    t0 = time.time()

    for npz_path in npz_files:
        result = train_one(npz_path, device=device)
        all_results.append(result)

    save_results(all_results, RESULTS_DIR)

    total_min = (time.time() - t0) / 60
    print(f"\nAll {len(npz_files)} IRs done in {total_min:.1f} min")
    print(f"Results → {os.path.abspath(RESULTS_DIR)}/")


if __name__ == "__main__":
    main()
