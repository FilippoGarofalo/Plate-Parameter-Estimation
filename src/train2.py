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
from torch.optim import Adam
from utils import load_challenge_npz
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lhs import lhs_sample_raw_params
from ground_truth import compute_nmse
import pandas as pd
from pathlib import Path


def main():
    # 1. SETUP & HYPERPARAMETERS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #target_npz_path = "target/ground_truth_random_42.npz"
    target_npz_path = "target/2026-DATASET-STRIPPED/random_IR_0009.npz" 
    sample_rate     = 44100
    num_iterations  = 1500
    LR              = 0.01
    dtype           = torch.float64

    # Multi-start settings
    n_starts        = 500   
    probe_iters     = 100   # short run per LHS start to find best basin
    lhs_seed        = 42

    ### MODIFIED: Bumped to 0.2 so 4096 and 8192 FFT sizes don't crash
    PHASE1_DURATION = 0.2  
    ### END MODIFIED ###

    target_stem  = Path(target_npz_path).stem
    target_index = target_stem.split('_')[-1] if '_' in target_stem else target_stem

    target_ir = load_challenge_npz(target_npz_path, device=device, dtype=dtype)

    # ... (il tuo codice di SETUP precedente) ...
    duration = len(target_ir) / sample_rate
    print(f"Target IR loaded: {len(target_ir)} samples ({duration:.2f} seconds)")

    criterion = Loss(
        mse_weight=0.0,
        stft_weight=1.0,
        energy_weight=0.0,
        fft_sizes=[64, 128, 256, 512, 1024, 2048, 4096],
    ).to(device)

    # ── PHASE 1: Multi-start Exploration (LHS) ────────────────
    print(f"\nPhase 1 — Multi-start exploration: {n_starts} starts, {probe_iters} iters each")
    
    # 1. Genera i sample raw usando la tua funzione LHS
    # Assumiamo che restituisca una lista di dizionari con i valori raw iniziali
    raw_samples = lhs_sample_raw_params(n_starts, seed=lhs_seed)
    best_loss = float('inf')
    best_raw_params = None

    rng = np.random.default_rng(lhs_seed)
    probe_loss_curves = {}  # {probe_index: [loss per iter]}
    
    probe_duration = 0.2 # 2205 campioni
    target_ir_cropped_probe = target_ir[:int(sample_rate * probe_duration)]
    
    # NOVITÀ: Creiamo un criterion dedicato al probe, SENZA fft_sizes enormi (che causano l'OOM!)
    # Il massimo qui è 1024, perfettamente contenuto in 2205 campioni.
    criterion_probe = Loss(
        mse_weight=0.0,
        stft_weight=1.0,
        energy_weight=0.0,
        fft_sizes=[256, 1024, 2048], 
    ).to(device)
    criterion_probe.precompute_target_stft(target_ir_cropped_probe)

    for i, init_params in enumerate(raw_samples):
        probe_model = DifferentiableModalPlate(
            sample_rate=sample_rate, 
            plate_params=init_params, 
            dtype=torch.float64
        ).to(device)
        
        probe_model.maxOm = 2500.0 * 2 * torch.pi
        probe_optimizer = Adam([
        {'params': [probe_model.Ly_raw, probe_model.xo_raw, probe_model.yo_raw], 'lr': 0.01},
            {
                'params': [probe_model.D_over_mu_raw, probe_model.T0_over_mu_raw], 
                'lr': 0.05
            },
            {
                'params': [probe_model.mu_raw], 
                'lr': 0.1
            }
        ])
        
        track_this_probe = i in plot_probe_indices
        if track_this_probe:
            probe_loss_curves[i] = []

        for _ in range(probe_iters):
            probe_optimizer.zero_grad(set_to_none=True)

            pred_ir = probe_model(duration=probe_duration, normalize=False, velCalc=False)
            loss = criterion_probe(pred_ir, target_ir_cropped_probe)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe_model.parameters(), max_norm=1.0)
            probe_optimizer.step()

            if track_this_probe:
                probe_loss_curves[i].append(loss.item())

        final_probe_loss = loss.item()
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Probe {i+1:03d}/{n_starts} | Loss finale: {final_probe_loss:.4f}")
            
        if final_probe_loss < best_loss:
            best_loss = final_probe_loss
            best_raw_params = {
                name: param.detach().cpu().item()
                for name, param in probe_model.named_parameters() if param.requires_grad
            }

        del probe_model
        del probe_optimizer
        del pred_ir
        del loss
        torch.cuda.empty_cache()

        if best_loss < 0.5:
            print(f"  Early stop: probe {i+1} reached loss {best_loss:.4f} < 0.5, skipping remaining probes")
            break

    print(f"\n>>> Miglior loss trovata in Phase 1: {best_loss:.4f}")
    print(">>> Parametri vincitori inizializzati per la Phase 2.")

    if probe_loss_curves:
        fig, ax = plt.subplots(figsize=(8, 4))
        for probe_idx, losses in probe_loss_curves.items():
            ax.plot(losses, label=f"Probe {probe_idx}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Phase 1 — sample probe loss curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        probe_plot_path = Path("experiment_results_taskA") / f"phase1_probe_curves_{target_index}.png"
        probe_plot_path.parent.mkdir(exist_ok=True)
        fig.savefig(probe_plot_path, dpi=150)
        plt.close(fig)
        print(f"Probe loss curves saved to {probe_plot_path}")
    else:
        print("No probe curves to plot (early stop before any tracked probe was reached)")
    # ──────────────────────────────────────────────────────────


    # ── PHASE 2: Full optimization from best start ────────────────
    print(f"\nPhase 2 — full optimization for {num_iterations} iterations from best start")

    # Inizializza il modello finale SOLO UNA VOLTA usando best_raw_params
    model = DifferentiableModalPlate(
        sample_rate=sample_rate,
        plate_params=best_raw_params, 
        dtype=dtype
    ).to(device)

    active_params = filter(lambda p: p.requires_grad, model.parameters())
    
    # Dividiamo i parametri dando una forza di rientro (weight_decay) a T0 e D
    optimizer = Adam([
        {'params': [model.mu_raw, model.Ly_raw, model.xo_raw, model.yo_raw], 'lr': 0.01},
        {
            'params': [model.D_over_mu_raw, model.T0_over_mu_raw], 
            'lr': 0.01
        }
    ])

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=500, min_lr=1e-4)
    
    progress = {'iteration': [], 'loss': [], 'mu': [], 'D_over_mu': [], 'T0_over_mu': [], 'Ly': [], 'xo': [], 'yo': []}

    STFT_DURATION = 3.0
    best_loss_phase2   = float('inf')
    best_params_phase2 = None

    # 3. OPTIMIZATION LOOP
    print("\nStarting Optimization")
    start_time = time.time()
    idx = -1
    for iteration in range(num_iterations):
        idx += 1
        optimizer.zero_grad()

        # Step 2: Forward Pass
        if iteration == 0: 
            print(" [diag] forward...", flush=True)

        ### MODIFIED: Restored the correct curriculum logic ###
        if iteration < 1000:
            curr_duration = min(0.05 + (iteration/2000)*STFT_DURATION, STFT_DURATION)
        else:
            curr_duration = STFT_DURATION

        pred_ir = model(duration=curr_duration, normalize=False, velCalc=False)
        curr_samples = pred_ir.shape[0]
        target_ir_cropped = target_ir[:curr_samples]
        criterion.precompute_target_stft(target_ir_cropped)

        loss = criterion(pred_ir, target_ir_cropped)

        if iteration == 0: 
            print(" [diag] loss...", flush=True)
            print(f" [diag] loss={loss.item():.6f} backward...", flush=True)

        
        scheduler.step(loss.item())
        if iteration% 10 == 0:
            print(f" [diag] iter {iteration}, loss={loss.item():.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")
            
        # Step 4: Backward Pass
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        if iteration == 0:
            grad_norms = {n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None}
            print(f" [diag] grad norms: {grad_norms}", flush=True)
        
        # Step 6: Update Parameters
        optimizer.step()
        
        current_loss_val = loss.item()

        # Track best physical parameters only after full-duration phase starts (iteration >= 1000)
        if iteration >= 1000 and current_loss_val < best_loss_phase2:
            best_loss_phase2 = current_loss_val
            with torch.no_grad():
                _mu, _D, _T0, _Ly, _xo, _yo = model.get_physical_parameters()
                best_params_phase2 = {
                    'mu':        _mu.item(),
                    'D_over_mu': _D.item(),
                    'T0_over_mu':_T0.item(),
                    'Ly':        _Ly.item(),
                    'xo':        _xo.item(),
                    'yo':        _yo.item(),
                }

        optimizer.zero_grad(set_to_none=True)
        del pred_ir
        del target_ir_cropped
        del loss # Safe to delete the tensor now
        gc.collect()
        torch.cuda.empty_cache()
        
        # Step 7: Print logs and parameter progress
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
    print(f"\nOptimization complete in {total_time:.2f} seconds.")

    np.savez('target/train_progress.npz', **{k: np.array(v) for k, v in progress.items()})
    print("Training progress saved to target/train_progress.npz")

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
    data_keys = npz_data.files
    overall_nmse = None
    if 'gt_mu' in data_keys:
        estimated = {
            'mu':         mu,
            'D_over_mu':  D_over_mu,
            'T0_over_mu': T0_over_mu,
            'Ly':         Ly,
            'xo':         xo,
            'yo':         yo,
        }
        _, overall_nmse = compute_nmse(estimated, target_npz_path)
    else:
        print("(NMSE skipped: target file has no embedded ground truth params)")

    # ── Save results to experiment_results_taskA/ ────────────────
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

    # Decide whether this run improves on any previous run for the same IR
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
        pd.DataFrame([best_params]).to_csv(output_path / f"best_params_{target_index}.csv", index=False, float_format='%.17g')
        print(f"\nResults saved to {summary_file}")



if __name__ == "__main__":
    main()