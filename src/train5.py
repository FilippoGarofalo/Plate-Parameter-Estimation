import torch
import time
import numpy as np
import scipy.signal as signal  ### MODIFIED: Added for target analysis ###
from model import DifferentiableModalPlate
from loss import Loss
from loss2 import MSELoss
from utils import load_challenge_npz
from optimizer import get_optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

def main():
    # 1. SETUP & HYPERPARAMETERS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    target_npz_path = "target/ground_truth_test_1.1.npz"
    #target_npz_path = "target/2026-DATASET-STRIPPED/random_IR_0001.npz"
    sample_rate = 44100
    num_iterations = 2500
    LR = 0.1
    dtype = torch.float64

    target_ir = load_challenge_npz(target_npz_path, device=device, dtype=dtype)

    duration = len(target_ir) / sample_rate
    print(f"Target IR loaded: {len(target_ir)} samples ({duration:.2f} seconds)")

    ### MODIFIED: Smart Initialization (Target Analysis) ###
    # Find the fundamental frequency of the target to guide parameter initialization
    ir_cpu = target_ir.cpu().numpy().flatten()
    ir_cpu = ir_cpu / (np.max(np.abs(ir_cpu)) + 1e-12) # normalize
    f_arr, Pxx = signal.welch(ir_cpu, fs=sample_rate, nperseg=16384)
    Pxx_db = 10 * np.log10(Pxx + 1e-12)
    
    # Ignore DC offset noise below 20Hz
    valid_idx = np.where(f_arr > 20)[0] 
    peaks, _ = signal.find_peaks(Pxx_db[valid_idx], prominence=10)
    
    if len(peaks) > 0:
        target_f0 = f_arr[valid_idx[peaks[0]]]
        print(f"Detected Target Fundamental: {target_f0:.2f} Hz")
    else:
        target_f0 = 100.0 # fallback
    ### END MODIFIED ###

    model = DifferentiableModalPlate(
        sample_rate=sample_rate,
        dtype=dtype
    ).to(device)
    
    ### MODIFIED: Smart Initialization (Parameter Nudging) ###
    # Nudge the starting parameters to be in the right "ballpark" for this specific file.
    # (Note: adjust the variable names `.D_over_mu` and `.T0_over_mu` if your raw nn.Parameters 
    # are named differently inside DifferentiableModalPlate)
    with torch.no_grad():
        if hasattr(model, 'D_over_mu') and hasattr(model, 'T0_over_mu'):
            if target_f0 > 100:
                model.D_over_mu.data.fill_(10.0)
                model.T0_over_mu.data.fill_(1000.0)
            else:
                model.D_over_mu.data.fill_(1.0)
                model.T0_over_mu.data.fill_(100.0)
    ### END MODIFIED ###

    criterion = Loss(
        mse_weight=0.0,
        stft_weight=1.0,
        energy_weight=0.0,
        ### MODIFIED: Added 8192 and 16384 for high-res low frequency mode detection
        fft_sizes=[64, 128, 256, 1024, 4096, 8192, 16384],
        ### END MODIFIED ###
       ).to(device)
    
    criterion2 = MSELoss().to(device)

    active_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = get_optimizer(active_params ,lr=LR)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, min_lr=1e-4)
    previous_lr = LR
    

    #criterion = MSELoss().to(device)
    progress = {'iteration': [], 'loss': [], 'mu': [], 'D_over_mu': [], 'T0_over_mu': [], 'Ly': [], 'xo': [], 'yo': []}

    # Before the loop, define constants:
    
    ### MODIFIED: Durations ###
    STFT_DURATION = 0.5     # fixed short window for STFT phase (must be >= 0.38s to fit 16384 FFT)
    MSE_DURATION = duration - 0.1 # dynamically stop 0.1s before the file ends to avoid the truncation cliff
    ### END MODIFIED ###
    
    use_mse = False
    mse_start_iter = None         # track when MSE phase begins

    # 3. OPTIMIZATION LOOP
    print("\nStarting Optimization")
    start_time = time.time()
    idx = -1
    for iteration in range(num_iterations):
        idx += 1
        # Step 1: Clear the gradients
        optimizer.zero_grad()

        # Step 2: Forward Pass
        if iteration == 0: 
            print(" [diag] forward...", flush=True)

        ### MODIFIED: Curriculum logic fixed ###
        if not use_mse:
            # Starts at 0.5s so the 16384 window doesn't crash, grows towards STFT_DURATION
            curr_duration = min(0.5 + (iteration/500)*STFT_DURATION, STFT_DURATION)
        else:
            mse_iters_elapsed = iteration - mse_start_iter
            # Seamlessly grow from STFT_DURATION up to MSE_DURATION (instead of dropping back to 0.05)
            curr_duration = min(STFT_DURATION + (mse_iters_elapsed / 500) * (MSE_DURATION - STFT_DURATION), MSE_DURATION)
        ### END MODIFIED ###

        pred_ir = model(duration=curr_duration, normalize=False, velCalc=False)
        curr_samples = pred_ir.shape[0]
        target_ir_cropped = target_ir[:curr_samples]
        #
        if not use_mse:
            criterion.precompute_target_stft(target_ir_cropped)
            loss = criterion(pred_ir, target_ir_cropped)
        else:
            loss = criterion2(pred_ir, target_ir_cropped)

        if iteration == 0: 
            print(" [diag] loss...", flush=True)
        #loss = criterion(pred_ir, target_ir)

        # Step 4: Backward Pass
        if iteration == 0: 
            print(f" [diag] loss={loss.item():.6f} backward...", flush=True)
        loss.backward()
        
        if iteration == 0:
            grad_norms = {n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None}
            print(f" [diag] grad norms: {grad_norms}", flush=True)

        
        # Step 6: Update Parameters
        optimizer.step()
        if not use_mse and loss.item() < 0.40:
            use_mse = True
            mse_start_iter = iteration
            optimizer.param_groups[0]['lr'] = 0.01
            print(f" [switch] → MSE at iter {iteration}, loss={loss.item():.4f}")


        optimizer.zero_grad()
        
        #if use_mse and curr_duration == MSE_DURATION:
        #    scheduler.step(loss.item())
        #    print(f" [diag] MSE phase: Plateau check at iter {iteration}, loss={loss.item():.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")
        
        # Otherwise, the growing signal duration will artificially trigger learning rate drops
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