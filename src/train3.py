import torch
import time
import numpy as np
from model import DifferentiableModalPlate
from loss import Loss
from loss2 import MSELoss
from utils import load_challenge_npz, map_sigm_linear, map_softplus_log, inverse_map_sigm_linear, inverse_map_softplus_log
from optimizer import get_optimizer
from scipy.stats import qmc
from torch.optim.lr_scheduler import ReduceLROnPlateau

def find_best_lhs_start(model, target_ir, criterion, device, dtype, num_samples=50):
    print(f"\n--- Ricerca del miglior punto di partenza via Latin Hypercube Sampling ({num_samples} samples) ---")
    
    sampler = qmc.LatinHypercube(d=4) 
    lhs_samples = sampler.random(n=num_samples)

    best_loss = float('inf')
    best_params = None
    
    test_duration = 0.05
    test_samples = int(44100 * test_duration)
    target_cropped = target_ir[:test_samples]
    
    # --- FIX CRITICO 1: Normalizziamo il target per la ricerca LHS ---
    peak_t = torch.max(torch.abs(target_cropped)) + 1e-8
    target_ir_norm = target_cropped / peak_t
    criterion.precompute_target_stft(target_ir_norm)

    for i in range(num_samples):
        # 1. Mappatura
        ly_phys = 1.1 + lhs_samples[i, 0] * (4.0 - 1.1)
        mu_phys = np.exp(np.log(2.43) + lhs_samples[i, 1] * (np.log(106.15) - np.log(2.43)))
        d_mu_phys = np.exp(np.log(0.2805) + lhs_samples[i, 2] * (np.log(201.188) - np.log(0.2805)))
        t0_mu_phys = np.exp(np.log(9.4e-5) + lhs_samples[i, 3] * (np.log(411.52) - np.log(9.4e-5)))

        # 2. Inverse (Controlla che il weight di T0_over_mu sia uguale a quello in model.py!)
        ly_raw = inverse_map_sigm_linear(ly_phys, 1.1, 4.0)
        mu_raw = inverse_map_softplus_log(mu_phys, 2.43, 106.15)
        d_mu_raw = inverse_map_softplus_log(d_mu_phys, 0.2805, 201.188)
        t0_mu_raw = inverse_map_softplus_log(t0_mu_phys, 9.4e-5, 411.52, weight=0.1) 

        # 3. Iniezione temporanea
        with torch.no_grad():
            model.Ly_raw.copy_(torch.tensor(ly_raw, dtype=dtype, device=device))
            model.mu_raw.copy_(torch.tensor(mu_raw, dtype=dtype, device=device))
            model.D_over_mu_raw.copy_(torch.tensor(d_mu_raw, dtype=dtype, device=device))
            model.T0_over_mu_raw.copy_(torch.tensor(t0_mu_raw, dtype=dtype, device=device))

            # --- FIX CRITICO 2: normalize=True per far agire il Centroide sullo spettro puro ---
            pred_ir = model(duration=test_duration, normalize=True, velCalc=False)
            
            if pred_ir.shape[0] < test_samples:
                pred_ir = torch.nn.functional.pad(pred_ir, (0, test_samples - pred_ir.shape[0]))
            
            # Calcolo loss col target normalizzato!
            loss = criterion(pred_ir, target_ir_norm)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = (ly_raw, mu_raw, d_mu_raw, t0_mu_raw)

    # 5. Applica definitivamente
    with torch.no_grad():
        model.Ly_raw.copy_(torch.tensor(best_params[0], dtype=dtype, device=device))
        model.mu_raw.copy_(torch.tensor(best_params[1], dtype=dtype, device=device))
        model.D_over_mu_raw.copy_(torch.tensor(best_params[2], dtype=dtype, device=device))
        model.T0_over_mu_raw.copy_(torch.tensor(best_params[3], dtype=dtype, device=device))

    print(f"LHS completato! Loss Iniziale crollata a: {best_loss:.4f}")
    print("-" * 60)

def main():
    # 1. SETUP & HYPERPARAMETERS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    target_npz_path = "target/ground_truth_test_1.2.npz"
    sample_rate = 44100
    num_iterations = 2500
    LR = 0.01
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
        fft_sizes=[16, 32, 64, 128, 256, 1024, 4096],
       ).to(device)
    
    criterion2 = MSELoss().to(device)

    # Note: Ly is NOT frozen here. Only spatial variables are.
    model.xo_raw.requires_grad = False
    model.yo_raw.requires_grad = False
    active_params = filter(lambda p: p.requires_grad, model.parameters())

    # Perform Warm-Start
    find_best_lhs_start(model, target_ir, criterion, device, dtype, num_samples=50)

    # Re-initialize optimizer so it picks up the copied weights
    optimizer = get_optimizer(active_params ,lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, min_lr=1e-4)

    progress = {'iteration': [], 'loss': [], 'mu': [], 'D_over_mu': [], 'T0_over_mu': [], 'Ly': [], 'xo': [], 'yo': []}

    print("\nStarting Optimization")
    start_time = time.time()
    idx = -1
    
    for iteration in range(num_iterations):
        idx += 1
        optimizer.zero_grad()

        if iteration == 0: 
            print(" [diag] forward...", flush=True)

        curr_duration = min(0.1 + (idx/500)*duration, duration) 

        # ========================================================
        # PHASE 1: NORMALIZED STFT (Match Spectral Shape)
        # ========================================================
        if iteration < 500:
            if iteration < 150:
                # Fase di stabilizzazione: manteniamo i 50ms dell'LHS
                curr_duration = 0.05
            else:
                # Dalla 150 alla 500, cresciamo dolcemente fino alla durata totale
                time_frac = (iteration - 150) / 350.0 
                curr_duration = min(0.05 + time_frac * (duration - 0.05), duration)
            # CRITICAL: normalize=True 
            pred_ir = model(duration=curr_duration, normalize=True, velCalc=False)
            curr_samples = pred_ir.shape[0]
            target_ir_cropped = target_ir[:curr_samples]
            
            # CRITICAL: Normalize the target as well
            peak_t = torch.max(torch.abs(target_ir_cropped)) + 1e-8
            target_ir_norm = target_ir_cropped / peak_t
            
            criterion.precompute_target_stft(target_ir_norm)
            loss = criterion(pred_ir, target_ir_norm)
            
        # ========================================================
        # PHASE 2: TIME DOMAIN MSE (Match Volume and Space)
        # ========================================================
        else:
            # ========================================================
        # FASE 1: NORMALIZED STFT (Allineamento Spettrale)
        # ========================================================
        if iteration < 800:
            if iteration < 150:
                # Stabilizzazione a 50ms
                curr_duration = 0.05
            else:
                # Crescita dolce
                time_frac = (iteration - 150) / 650.0 
                curr_duration = min(0.05 + time_frac * (duration - 0.05), duration)
                
            pred_ir = model(duration=curr_duration, normalize=True, velCalc=False)
            curr_samples = pred_ir.shape[0]
            target_ir_cropped = target_ir[:curr_samples]
            
            peak_t = torch.max(torch.abs(target_ir_cropped)) + 1e-8
            target_ir_norm = target_ir_cropped / peak_t
            
            criterion.precompute_target_stft(target_ir_norm)
            loss = criterion(pred_ir, target_ir_norm)
            
        # ========================================================
        # FASE 2: UNNORMALIZED STFT + ENERGY (Allineamento Volume/Spazio)
        # ========================================================
        else:
            if iteration == 800:
                print("\n [switch] Fase 2: Sblocco spaziali, tolgo normalizzazione e attivo Energy Loss!", flush=True)
                model.xo_raw.requires_grad = True
                model.yo_raw.requires_grad = True
                optimizer.add_param_group({'params': [model.xo_raw, model.yo_raw], 'lr': 0.01})
                
                # LA MAGIA È QUI: Accendiamo l'Energy Loss all'interno del criterio STFT
                criterion.energy_weight = 1.0
                
            pred_ir = model(duration=curr_duration, normalize=False, velCalc=False)
            curr_samples = pred_ir.shape[0]
            target_ir_cropped = target_ir[:curr_samples]
            
            # Ricalcoliamo le STFT target ma questa volta SENZA normalizzarle
            criterion.precompute_target_stft(target_ir_cropped)
            
            # Usiamo sempre criterion (STFT + Energy + Centroid). Addio MSE nel tempo!
            loss = criterion(pred_ir, target_ir_cropped)

        # --------------------------------------------------------
        # BACKWARD PASS
        # --------------------------------------------------------
        if iteration == 0: 
            print(" [diag] loss...", flush=True)
            print(f" [diag] loss={loss.item():.6f} backward...", flush=True)

        loss.backward()
        
        if iteration == 0:
            grad_norms = {n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None}
            print(f" [diag] grad norms: {grad_norms}", flush=True)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        
        if iteration % 10 == 0 or iteration == num_iterations - 1:
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

if __name__ == "__main__":
    main()