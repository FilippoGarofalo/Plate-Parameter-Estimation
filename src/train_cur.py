import argparse
import torch
import time
import copy
import numpy as np
from model import DifferentiableModalPlate
from loss import Loss
from utils import load_challenge_npz
from optimizer import get_optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lhs import lhs_sample_raw_params_2d

def train_on_target(
    target_npz_path,
    num_iterations=1000,
    print_every=100,
    progress_path="target/train_progress.npz",
):
    # 1. SETUP & HYPERPARAMETERS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sample_rate = 44100
    LR = 0.1
    dtype = torch.float64
    '''
    # Multi-start settings
    n_starts = 100
    lhs_seed = 42
    '''
    # Bumped to 0.2 so 4096 and 8192 FFT sizes don't crash
    PHASE1_DURATION = 0.2

    target_ir = load_challenge_npz(target_npz_path, device=device, dtype=dtype)

    duration = len(target_ir) / sample_rate
    print(f"Target IR loaded: {len(target_ir)} samples ({duration:.2f} seconds)")

    # Loss parameters kept as requested
    criterion = Loss(
        mse_weight=0.0,
        stft_weight=1.0,
        energy_weight=0.0,
        fft_sizes=[64, 128, 256, 1024, 4096]
    ).to(device)

    '''
    # ── PHASE 1: ZERO-SHOT PROBING ────────────────────────────
    lhs_params = lhs_sample_raw_params_2d(n_starts, seed=lhs_seed)
    print(f"\nPhase 1 — Zero-shot probing {n_starts} LHS starts (Ultra-fast, No Gradients)")

    best_probe_loss  = float('inf')
    best_state_dict  = None
    phase1_start     = time.time()

    probe_target = target_ir[:int(PHASE1_DURATION * sample_rate)]
    criterion.precompute_target_stft(probe_target)

    for start_idx, raw_params in enumerate(lhs_params):
        model = DifferentiableModalPlate(
            sample_rate=sample_rate,
            plate_params=raw_params,
            dtype=dtype
        ).to(device)

        with torch.no_grad():
            pred_ir = model(duration=PHASE1_DURATION, normalize=False, velCalc=False)
            probe_loss = criterion(pred_ir, probe_target).item()

        mu, D_over_mu, T0_over_mu, Ly, xo, yo = [
            p.cpu().item() for p in model.get_physical_parameters()
        ]
        
        if probe_loss < best_probe_loss:
            best_probe_loss = probe_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            
            print(f"  Start {start_idx + 1:02d}/{n_starts} | probe loss: {probe_loss:.6f} | "
                  f"Ly: {Ly:.4f}m | xo: {xo:.4f}m | yo: {yo:.4f}m | "
                  f"mu: {mu:.4f} | D/mu: {D_over_mu:.6f} | T0/mu: {T0_over_mu:.6f}")
            print(f"  >> New best probe (loss={best_probe_loss:.6f})")

        del model, pred_ir
        torch.cuda.empty_cache()

    phase1_time = time.time() - phase1_start
    print(f"\nPhase 1 done in {phase1_time:.2f}s. Best probe loss: {best_probe_loss:.6f}")

    '''
    # ── PHASE 2: CURRICULUM OPTIMIZATION ──────────────────────────────────────────
    print(f"\nPhase 2 — Curriculum Optimization for {num_iterations} iterations from best start")

    model = DifferentiableModalPlate(
        sample_rate=sample_rate,
        dtype=dtype
    ).to(device)
    #model.load_state_dict(best_state_dict)

    # CURRICULUM STEP 1: Freeze positional geometry (xo, yo)
    # We want the optimizer to focus solely on frequencies (materials & dimensions) first.
    model.xo_raw.requires_grad = False
    model.yo_raw.requires_grad = False
    print("\n[Curriculum Phase 2A] Freezing 'xo' and 'yo'. Optimizing Materials & Size (Frequencies).")

    active_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_optimizer(active_params, lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-4)
    
    progress = {'iteration': [], 'loss': [], 'mu': [], 'D_over_mu': [], 'T0_over_mu': [], 'Ly': [], 'xo': [], 'yo': []}

    STFT_DURATION = 2.0
    
    # We switch the curriculum exactly halfway through the iterations
    CURRICULUM_SWITCH_ITER = num_iterations // 2 

    # 3. OPTIMIZATION LOOP
    start_time = time.time()
    last_loss = None
    
    for iteration in range(num_iterations):
        
        # --- CURRICULUM SWITCH POINT ---
        if iteration == CURRICULUM_SWITCH_ITER:
            print("\n" + "="*60)
            print("[Curriculum Phase 2B] Unfreezing 'xo' and 'yo'. Optimizing Full Geometry (Amplitudes/Nodes).")
            print("="*60 + "\n")
            
            # Scongela le posizioni
            model.xo_raw.requires_grad = True
            model.yo_raw.requires_grad = True
            
            # Re-inizializza l'ottimizzatore per includere i nuovi parametri, con LR ridotto
            # Questo resetta anche il momento di Adam, utile per non sovrascrivere le nuove variabili
            active_params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = get_optimizer(active_params, lr=LR * 0.1) 
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-5)
        # -------------------------------

        optimizer.zero_grad()

        # Assicuriamoci di non superare mai la durata reale del target
        max_target_duration = target_ir.shape[0] / sample_rate
        target_max_stft = min(STFT_DURATION, max_target_duration)
        
        # Rampa dinamica che arriva al massimo prima del cambio di curriculum
        ramp_factor = min(1.0, iteration / (CURRICULUM_SWITCH_ITER * 0.8))
        curr_duration = min(0.05 + ramp_factor * target_max_stft, target_max_stft)

        pred_ir = model(duration=curr_duration, normalize=False, velCalc=False)
        
        # Pareggiamo strettamente le dimensioni (evita crash se il target è più corto di 2 secondi)
        min_samples = min(pred_ir.shape[0], target_ir.shape[0])
        pred_ir = pred_ir[:min_samples]
        target_ir_cropped = target_ir[:min_samples]

        criterion.precompute_target_stft(target_ir_cropped)
        loss = criterion(pred_ir, target_ir_cropped)

        # Step 4: Backward Pass
        loss.backward()

        # Step 6: Update Parameters
        optimizer.step()
        
        if curr_duration >= STFT_DURATION:
            scheduler.step(loss.item())
            
        # Logging
        if iteration % print_every == 0 or iteration == num_iterations - 1:
            mu, D_over_mu, T0_over_mu, Ly, xo, yo = [
                p.detach().cpu().item() for p in model.get_physical_parameters()
            ]
            loss_val = loss.item()
            last_loss = loss_val
            loss_str = f"{loss_val:.2e}" if loss_val < 1e-3 else f"{loss_val:.6f}"
            
            phase_tag = "2A (Freqs)" if iteration < CURRICULUM_SWITCH_ITER else "2B (Amps)"
            print(f"Iter {iteration:04d} [{phase_tag}] | Loss: {loss_str} | LR: {optimizer.param_groups[0]['lr']:.6f}")
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

    if progress_path:
        np.savez(progress_path, **{k: np.array(v) for k, v in progress.items()})
        print(f"Training progress saved to {progress_path}")

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

    result = {
        "loss": last_loss if last_loss is not None else loss.item(),
        "params": {
            "mu": mu,
            "D_over_mu": D_over_mu,
            "T0_over_mu": T0_over_mu,
            "Ly": Ly,
            "xo": xo,
            "yo": yo,
        },
    }

    del model, criterion, optimizer, scheduler, target_ir, probe_target
    torch.cuda.empty_cache()
    return result

def main():
    parser = argparse.ArgumentParser(description="Plate parameter estimation")
    parser.add_argument("target_npz", type=str, help="Path to target .npz IR file")
    parser.add_argument("--print-every", type=int, default=100, help="Log every N iterations")
    parser.add_argument("--num-iterations", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--progress-path", type=str, default="target/train_progress.npz")
    args = parser.parse_args()

    train_on_target(
        args.target_npz,
        num_iterations=args.num_iterations,
        print_every=args.print_every,
        progress_path=args.progress_path,
    )

if __name__ == "__main__":
    main()