import torch
import time
from model import DifferentiableModalPlate
from loss import TimeDomainEnergyLoss
from utils import load_challenge_npz, invert_composite_parameters
from optimizer import get_optimizer

def main():
    # 1. SETUP & HYPERPARAMETERS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Path del target non normalizzato[cite: 11]
    target_npz_path = "target/ground_truth_test.npz"
    sample_rate = 44100
    num_iterations = 2000
    LR = 0.1 
    dtype = torch.float32

    # Caricamento Target RAW (non normalizzato)[cite: 11, 19]
    target_ir_raw = load_challenge_npz(target_npz_path, device=device, dtype=dtype)
    
    # Creazione manuale del Target NORMALIZED (seguendo la tua sintassi)[cite: 11, 23]
    target_peak = torch.max(torch.abs(target_ir_raw)) + 1e-8
    target_ir_norm = target_ir_raw / target_peak

    duration = len(target_ir_raw) / sample_rate
    print(f"Target IR loaded: {len(target_ir_raw)} samples ({duration:.2f} seconds)")

    # 2. INITIALIZE MODULES[cite: 7, 15, 23]
    model = DifferentiableModalPlate(sample_rate=sample_rate, plate_params=None, dtype=dtype).to(device)
    
    # Configurazione pesi: STFT per la geometria, Energy per mu e materiali[cite: 9, 17]
    criterion = TimeDomainEnergyLoss(
        mse_weight=0.0, 
        stft_weight=5.0,      # Peso per Ly, xo, yo (su segnale normalizzato)
        energy_weight=0.0,     # Peso per mu, D/mu, T0/mu (su segnale non normalizzato)
        lowpass_weight=0.0,    
        fft_sizes=[64, 256, 1024, 4096]
    ).to(device)

    active_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_optimizer(active_params, lr=LR)

    # Scheduler per stabilizzare la convergenza finale[cite: 6, 22]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30, verbose=True
    )

    # 3. OPTIMIZATION LOOP
    print("\nStarting Optimization (Hybrid Normalized/Raw strategy)")
    start_time = time.time()
    
    for iteration in range(num_iterations):
        # Rimuovi i da qui:
        optimizer.zero_grad() 

        # STEP 2: Generazione doppia (Normalizzata e Raw)
        pred_ir_norm = model(duration=duration, normalize=True, velCalc=False)
        pred_ir_raw  = model(duration=duration, normalize=False, velCalc=False)
        
        # STEP 3: Calcolo Loss differenziate
        loss_geo = criterion.stft_weight * criterion(pred_ir_norm, target_ir_norm)
        
        pred_energy = pred_ir_raw ** 2
        target_energy = target_ir_raw ** 2
        loss_phys = criterion.energy_weight * torch.mean((pred_energy - target_energy) ** 2) / (torch.mean(target_energy**2) + 1e-8)

        total_loss = loss_geo + loss_phys

        # STEP 4 & 5: Backward e Clipping
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

        optimizer.step()
        scheduler.step(total_loss)

        # STEP 7: Logging[cite: 6, 14, 22]
        if iteration % 10 == 0 or iteration == num_iterations - 1:
            mu, D_over_mu, T0_over_mu, Ly, xo, yo = [
                p.detach().cpu().item() for p in model.get_physical_parameters()
            ]
            
            print(f"Iteration {iteration:04d} | Total Loss: {total_loss.item():.4f} (Geo: {loss_geo.item():.4f}, Phys: {loss_phys.item():.4f})")
            print(f"Ly: {Ly:.4f}m | xo: {xo:.4f}m | yo: {yo:.4f}m | "
                  f"mu: {mu:.4f} | D/mu: {D_over_mu:.6f} | T0/mu: {T0_over_mu:.6f}")
            print("-" * 60)

    total_time = time.time() - start_time
    print(f"\nOptimization complete in {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()